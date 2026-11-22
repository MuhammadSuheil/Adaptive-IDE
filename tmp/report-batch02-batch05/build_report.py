"""Create the meeting report from immutable result files; no model execution."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
DEST = ROOT / "models/reports/Laporan_Eksperimen_Batch_02_sampai_05.docx"
DATA = ROOT / "analysis_sessions/dataset_fusion/output/v2_session_phase"
RUNS = {
    (2, 1): "20261119T185251108907Z_2bc268ab",
    (2, 2): "20261119T191445996307Z_f399c8da",
    (3, 1): "20261120T032808093791Z_41844d37",
    (3, 2): "20261120T040350303624Z_6156e180",
    (4, 1): "20261121T025639248779Z_9443342741",
    (4, 2): "20261121T073428389875Z_8be6ef1c10",
    (4, 3): "20261121T090223169904Z_62687ffa6e",
}
STAGE2 = ROOT / "models/batch-05/outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829"
STAGE3 = ROOT / "models/batch-05/outputs/exp01/stage3_20261122T115353575899Z_933c7b4bcb"
CLOSING = ROOT / "models/batch-05/outputs/closing_audit"
SOURCES = set()


def read_json(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    SOURCES.add(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def percent(value):
    return f"{float(value) * 100:.2f}".replace(".", ",") + "%"


def number(value):
    return f"{int(value):,}".replace(",", ".")


RESULTS = {}
SIGNATURES = {}
SELECTIONS = {}
for key, run in RUNS.items():
    batch, experiment = key
    folder = ROOT / f"models/batch-{batch:02d}/outputs/exp{experiment:02d}" / run
    content = read_json(folder / "run.json")
    SIGNATURES[key] = content.get("signature", content)
    if batch in (2, 3):
        SELECTIONS[key] = read_json(folder / "selection/record.json")
    for row in read_csv(folder / "test_summary.csv"):
        if batch == 4 and row["window_winner"] != "True":
            continue
        RESULTS[batch, experiment, int(row["window"])] = row

OLD = read_json(DATA / "history/pre_feature_aware_manifest.json")
NEW = read_json(DATA / "manifest.json")
AUDIT = read_json(CLOSING / "audit_result.json")
TEST5 = read_json(STAGE3 / "test_reference.json")["metrics"]
SELECT5 = read_json(STAGE3 / "selection.json")
VAL5 = read_csv(STAGE3 / "validation_metrics.csv")
WINNER5 = next(r for r in VAL5 if r["selected"] == "True")
CORE9_ROWS = [
    r for r in read_csv(CLOSING / "core9_paired_differences.csv") if r["fold"] == "OOF"
]
CORE9 = {r["mode"]: r for r in CORE9_ROWS}
CV2 = read_csv(STAGE2 / "cv_leaderboard.csv")
LEARNING = read_csv(STAGE2 / "learning_curve.csv")

for w in (15, 25, 45):
    path = DATA / f"window_{w}s_step_5s/good_acceptable.csv"
    SOURCES.add(path)
    current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    old_hashes = {SIGNATURES[b, e]["dataset_sha256"][str(w)] for b, e in RUNS if b < 4}
    new_hashes = {SIGNATURES[4, e]["dataset_sha256"][str(w)] for e in (1, 2, 3)}
    if (
        len(old_hashes) != 1
        or new_hashes != {current_hash}
        or current_hash in old_hashes
    ):
        raise ValueError("Unexpected historical/current dataset identity")
    if (
        OLD["datasets"][f"window_{w}s_step_5s"]["good_acceptable"]["windows"]
        != {15: 3579, 25: 3214, 45: 2527}[w]
    ):
        raise ValueError("Historical source counts changed")
if (
    len(RESULTS) != 21
    or TEST5["rows"] != 683
    or AUDIT["trace"]["feature_mismatches"] != 0
):
    raise ValueError("Unexpected report evidence")

MODELS = {
    (2, 1, 15): "Soft voting LR interaksi + ET + RF",
    (2, 1, 25): "Logistic Regression fusion",
    (2, 1, 45): "Late fusion LR eye + RF HRV",
    (2, 2, 15): "MLP fusion 64 dan 32",
    (2, 2, 25): "Voting MLP + Transformer",
    (2, 2, 45): "Voting Transformer + residual MLP",
    (3, 1, 15): "HistGradientBoosting fusion",
    (3, 1, 25): "SVM RBF relative berbobot",
    (3, 1, 45): "Hard voting tiga konfigurasi",
    (3, 2, 15): "Residual MLP nonspatial 3 seed",
    (3, 2, 25): "Residual MLP nonspatial",
    (3, 2, 45): "Residual MLP nonspatial 3 seed",
    (4, 1, 15): "Stacking",
    (4, 1, 25): "SVM RBF core",
    (4, 1, 45): "CatBoost core",
    (4, 2, 15): "TabM core-head",
    (4, 2, 25): "Residual MLP core",
    (4, 2, 45): "TabM core-head 5 seed",
    (4, 3, 15): "Stacking",
    (4, 3, 25): "Hard voting",
    (4, 3, 45): "GRU core history 60 detik 5 seed",
}


def font(size, bold=False):
    return ImageFont.truetype(
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size
    )


def text_center(draw, xy, text, size=30, bold=False, fill="#222222"):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor="mm")


def make_figures():
    image = Image.new("RGB", (1700, 1040), "white")
    draw = ImageDraw.Draw(image)
    colors = ["#325A78", "#B87845", "#648477"]
    for index, batch in enumerate((2, 3, 4, 5)):
        left = 100 + (index % 2) * 840
        top = 95 + (index // 2) * 500
        width, height = 650, 270
        title = f"Batch {batch:02d}"
        if batch == 5:
            title += "  model terkunci"
        draw.text((left, top - 65), title, font=font(33, True), fill="black")
        for tick in (0, 20, 40, 60):
            y = top + height - height * tick / 60
            draw.line((left, y, left + width, y), fill="#D9D9D9", width=2)
            draw.text((left - 49, y - 16), str(tick), font=font(27), fill="#444444")
        experiments = [1, 2] if batch in (2, 3) else [1, 2, 3] if batch == 4 else [1]
        windows = [15, 25, 45] if batch != 5 else [25]
        for position, window in enumerate(windows):
            center = left + (position + 0.5) * width / len(windows)
            group_width = 156 if len(experiments) == 3 else 125
            for exp in experiments:
                val = (
                    float(RESULTS[batch, exp, window]["accuracy"])
                    if batch != 5
                    else TEST5["accuracy"]
                )
                bar_width = group_width / len(experiments) - 6
                x = (
                    center
                    - group_width / 2
                    + (exp - 1) * group_width / len(experiments)
                )
                y = top + height - height * val / 0.60
                draw.rectangle(
                    (x, y, x + bar_width, top + height), fill=colors[exp - 1]
                )
                text_center(
                    draw,
                    (x + bar_width / 2, y - 22),
                    f"{val * 100:.1f}".replace(".", ","),
                    25,
                    True,
                )
            text_center(draw, (center, top + height + 28), f"{window} detik", 29)
        if batch != 5:
            for exp in experiments:
                x = left + (exp - 1) * 205
                draw.rectangle(
                    (x, top + height + 70, x + 25, top + height + 94),
                    fill=colors[exp - 1],
                )
                draw.text(
                    (x + 38, top + height + 64),
                    f"Exp{exp:02d}",
                    font=font(28),
                    fill="black",
                )
        else:
            draw.text(
                (left, top + height + 68),
                "Accuracy Test referensi",
                font=font(28),
                fill="black",
            )
    image.save(WORK / "finalists.png")

    image = Image.new("RGB", (1600, 630), "white")
    draw = ImageDraw.Draw(image)
    x0, y0, width, height = 125, 82, 1300, 380
    draw.text(
        (x0, 15),
        "Batch 05  Learning curve Extra Trees session-robust",
        font=font(34, True),
        fill="black",
    )
    for tick in (0, 20, 40, 60, 80, 100):
        y = y0 + height - height * tick / 100
        draw.line((x0, y, x0 + width, y), fill="#DDDDDD", width=2)
        draw.text((x0 - 65, y - 18), str(tick), font=font(29), fill="#333333")
    for partition, color, label in [
        ("fitting", "#325A78", "Fitting"),
        ("validation", "#B87845", "Held-out fold"),
    ]:
        points = []
        for index, fraction in enumerate((0.25, 0.5, 0.75, 1.0)):
            values = [
                float(r["macro_f1"])
                for r in LEARNING
                if r["partition"] == partition and float(r["fraction"]) == fraction
            ]
            value = float(np.mean(values))
            x, y = x0 + 130 + index * 345, y0 + height - height * value
            points.append((x, y))
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)
            text_center(
                draw,
                (x, y - 25),
                f"{100 * value:.1f}".replace(".", ","),
                28,
                True,
                color,
            )
            if partition == "validation":
                text_center(draw, (x, y0 + height + 30), f"{int(fraction * 100)}%", 30)
        draw.line(points, fill=color, width=5)
        draw.rectangle(
            (
                x0 + (0 if partition == "fitting" else 440),
                555,
                x0 + 26 + (0 if partition == "fitting" else 440),
                581,
            ),
            fill=color,
        )
        draw.text(
            (x0 + 45 + (0 if partition == "fitting" else 440), 548),
            label,
            font=font(31),
            fill="black",
        )
    image.save(WORK / "learning_gap.png")


DOC = Document()
section = DOC.sections[0]
section.page_width, section.page_height = Inches(8.5), Inches(11)
section.top_margin, section.bottom_margin = Inches(0.78), Inches(0.72)
section.left_margin = section.right_margin = Inches(0.85)
section.footer_distance = Inches(0.3)
for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Caption"]:
    style = DOC.styles[name]
    style.font.name = "Arial"
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.font.bold = name in ("Title", "Heading 1", "Heading 2")
    style.font.italic = False
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Arial")
    style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        style._element.rPr.rFonts.attrib.pop(qn(f"w:{attr}"), None)
    properties = style._element.find(qn("w:pPr"))
    if properties is not None:
        for item in list(properties):
            if item.tag in {qn("w:pBdr"), qn("w:numPr")}:
                properties.remove(item)
DOC.styles["Normal"].font.size = Pt(11)
DOC.styles["Normal"].paragraph_format.line_spacing = 1.10
DOC.styles["Normal"].paragraph_format.space_after = Pt(6)
DOC.styles["Title"].font.size = Pt(22)
DOC.styles["Title"].paragraph_format.space_after = Pt(8)
DOC.styles["Subtitle"].font.size = Pt(12)
DOC.styles["Heading 1"].font.size = Pt(15)
DOC.styles["Heading 1"].paragraph_format.space_before = Pt(0)
DOC.styles["Heading 1"].paragraph_format.space_after = Pt(9)
DOC.styles["Heading 2"].font.size = Pt(11.5)
DOC.styles["Heading 2"].paragraph_format.space_before = Pt(8)
DOC.styles["Heading 2"].paragraph_format.space_after = Pt(5)
DOC.styles["Caption"].font.size = Pt(9.5)
DOC.styles["Caption"].font.italic = False
DOC.styles["Caption"].paragraph_format.space_after = Pt(6)
DOC.core_properties.title = (
    "Laporan Eksperimen Klasifikasi Beban Kognitif Batch 02 sampai 05"
)
DOC.core_properties.author = "Tim Penelitian"
DOC.core_properties.subject = (
    "Hasil model eye tracking dan HRV serta diagnosis empat batch eksperimen"
)
footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
run = footer.add_run()
run.font.name, run.font.size = "Arial", Pt(9)
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
run._r.addnext(field)
TABLE_INDEX = 0
FIGURE_INDEX = 0


def p(text, style=None):
    paragraph = DOC.add_paragraph(text, style)
    paragraph.paragraph_format.widow_control = True
    return paragraph


def h(text):
    return DOC.add_heading(text, 2)


def page(title):
    heading = DOC.add_heading(title, 1)
    heading.paragraph_format.page_break_before = True


def table(title, headers, rows, widths):
    global TABLE_INDEX
    TABLE_INDEX += 1
    caption = p(f"Tabel {TABLE_INDEX}. {title}", "Caption")
    caption.paragraph_format.keep_with_next = True
    tbl = DOC.add_table(rows=1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    for col, width in zip(tbl.columns, widths):
        col.width = Inches(width)
    properties = tbl._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        item = OxmlElement(f"w:{edge}")
        for key, value in [("val", "single"), ("sz", "4"), ("color", "D9D9D9")]:
            item.set(qn(f"w:{key}"), value)
        borders.append(item)
    properties.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for edge in ["top", "left", "bottom", "right"]:
        item = OxmlElement(f"w:{edge}")
        item.set(qn("w:w"), "80")
        item.set(qn("w:type"), "dxa")
        margins.append(item)
    properties.append(margins)
    repeat = OxmlElement("w:tblHeader")
    tbl.rows[0]._tr.get_or_add_trPr().append(repeat)
    for index, values in enumerate([headers, *rows]):
        row = tbl.rows[0] if index == 0 else tbl.add_row()
        cant = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(cant)
        for j, (cell, value) in enumerate(zip(row.cells, values)):
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = str(value)
            shade = OxmlElement("w:shd")
            shade.set(
                qn("w:fill"),
                "24445C" if index == 0 else "F2F5F7" if index % 2 else "FFFFFF",
            )
            cell._tc.get_or_add_tcPr().append(shade)
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(2)
            para.paragraph_format.space_before = Pt(2)
            para.paragraph_format.line_spacing = 1.05
            para.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if j == 1 and len(headers) > 3
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            for run in para.runs:
                run.font.name, run.font.size = "Arial", Pt(9.5)
                run.bold = index == 0
                run.font.color.rgb = (
                    RGBColor(255, 255, 255) if index == 0 else RGBColor(0, 0, 0)
                )
    spacer = p("")
    spacer.paragraph_format.space_after = Pt(1)
    spacer.paragraph_format.line_spacing = 0.25
    return tbl


def figure(path, caption, width=6.55):
    global FIGURE_INDEX
    FIGURE_INDEX += 1
    para = p("")
    para.paragraph_format.keep_with_next = True
    picture = para.add_run().add_picture(str(path), width=Inches(width))
    picture._inline.docPr.set("descr", caption)
    p(f"Gambar {FIGURE_INDEX}. {caption}", "Caption")


def result_table(batch):
    rows = []
    for b, e, w in sorted(k for k in RESULTS if k[0] == batch):
        r = RESULTS[b, e, w]
        val = r.get("selection_accuracy", r.get("validation_accuracy"))
        if val is None:
            val = next(
                candidate["validation"]["accuracy"]
                for candidate in SELECTIONS[b, e]["finalists"]
                if int(candidate["window"]) == w
            )
        model = MODELS[b, e, w] + (" *" if r["primary"] == "True" else "")
        rows.append(
            [
                f"{e:02d} / {w}s",
                model,
                percent(val),
                percent(r["accuracy"]),
                percent(r["macro_f1"]),
            ]
        )
    table(
        f"Finalist Batch {batch:02d}",
        [
            "Exp / window",
            "Model",
            "Accuracy validation",
            "Accuracy Test",
            "Macro-F1 Test",
        ],
        rows,
        [0.75, 2.7, 1.05, 1.05, 1.05],
    )
    p(
        "Tanda * menunjukkan model utama notebook yang dipilih melalui validation sebelum Test dibaca. Semua baris merupakan pemenang per-window, bukan seluruh kandidat pencarian.",
        "Caption",
    )


def build():
    make_figures()
    # Page 1
    p("Laporan Eksperimen Klasifikasi Beban Kognitif", "Title")
    p("Batch 02 sampai 05", "Subtitle")
    p("Status hasil per 22 November 2026", "Caption")
    h("Ringkasan")
    p(
        "Kami mengevaluasi klasifikasi Low, Medium, dan High menggunakan fitur eye tracking dan heart rate variability (HRV). Empat batch mencakup delapan notebook utama yang selesai, mulai dari model tabular dan neural hingga pemodelan urutan sensor serta kontrol diagnostik. Belum ada hasil akhir yang mencapai accuracy 70% pada interval waktu yang dipisahkan dari fitting."
    )
    top = [(2, 2, 15), (3, 1, 15), (4, 2, 45)]
    rows = [
        [
            f"{b:02d}",
            MODELS[b, e, w] + f", {w}s",
            percent(RESULTS[b, e, w]["accuracy"]),
            percent(RESULTS[b, e, w]["macro_f1"]),
        ]
        for b, e, w in top
    ]
    rows.append(
        [
            "05",
            "Extra Trees session-robust, 25s",
            percent(TEST5["accuracy"]),
            percent(TEST5["macro_f1"]),
        ]
    )
    table(
        "Hasil Test tertinggi yang teramati pada setiap batch",
        ["Batch", "Model", "Accuracy", "Macro-F1"],
        rows,
        [0.55, 3.65, 1.2, 1.2],
    )
    p(
        "Skor tertinggi yang teramati tidak selalu berasal dari model utama yang dipilih melalui validation. Batch 02 dan 03 menggunakan dataset historis yang sama; Batch 04 dan 05 memakai dataset setelah pembaruan kelayakan fitur. Pembagian development juga berubah. Karena itu, urutan angka pada tabel bukan pengukuran penurunan kemampuan algoritma dengan seluruh kondisi lain tetap."
    )
    p(
        "Eksperimen menemukan dan menangani masalah pipeline yang nyata, termasuk normalisasi antarsesi, kalibrasi SVM, refit, serta pemilihan model dan checkpoint. Setelah pembenahan tersebut, gap fitting-held-out tetap besar. Audit terakhir tidak menemukan salah hitung pada enam window yang ditelusuri ke sensor mentah, dan penghapusan satu fitur iris berulang tidak memberi peningkatan berarti."
    )
    p(
        "Kesimpulan untuk tim adalah bahwa kekurangan jumlah Low bukan satu-satunya penjelasan. Kelayakan teknis data telah diperiksa, tetapi kecukupan informasi sensor untuk membedakan label protokol belum terbukti. Laporan berikut merangkum tindakan, hasil, dan dasar diagnosis tersebut."
    )

    # Page 2
    page("1 Dataset dan pembentukan label")
    p(
        "Seluruh batch menggunakan tiga dataset good_acceptable dengan window 15, 25, dan 45 detik serta step 5 detik. Ketiganya merupakan alternatif representasi dari sumber rekaman yang sama, bukan tiga kelompok peserta independen yang ditumpuk menjadi satu dataset."
    )
    table(
        "Komposisi dataset yang digunakan dalam eksperimen",
        ["Pemakaian", "Window", "Low", "Medium", "High", "Total"],
        [
            ["Batch 02 dan 03", "15s", "1.062", "1.266", "1.251", "3.579"],
            ["Batch 02 dan 03", "25s", "984", "1.093", "1.137", "3.214"],
            ["Batch 02 dan 03", "45s", "908", "728", "891", "2.527"],
            ["Batch 04 dan 05", "15s", "1.687", "1.763", "1.803", "5.253"],
            ["Batch 04 dan 05", "25s", "1.549", "1.487", "1.572", "4.608"],
            ["Batch 04 dan 05", "45s", "1.394", "966", "1.149", "3.509"],
        ],
        [1.55, 0.7, 0.85, 0.85, 0.85, 1.8],
    )
    h("Satu window merangkum satu interval sensor")
    p(
        "Sebagai contoh, window 25 detik dapat merangkum detik 0 sampai 25, kemudian detik 5 sampai 30. Kedua baris berbagi 20 detik pengukuran. Jumlah baris karena itu tidak sama dengan jumlah observasi independen. Builder menghitung RMSSD dan SDNN dari RR mentah, lalu merangkum frame eye pada batas waktu absolut yang sama. Paket RR yang belum tersedia pada akhir window tidak dimasukkan secara retroaktif."
    )
    h("Label mengikuti difficulty dan fase yang terlacak")
    p(
        "Pengerjaan aktif easy, practice, dan demo diberi Low; pengerjaan medium diberi Medium; hard diberi High. Fase pre-task setelah kalibrasi, feedback, rating, rest, post-task, dan interrupted yang terlacak juga diberi Low. Kalibrasi dan waktu tanpa konteks yang dapat ditentukan tidak otomatis diberi Low."
    )
    p(
        "Seluruh bagian sebuah window harus menghasilkan satu target. Campuran easy dan feedback dapat tetap Low. Campuran medium dan rating dikeluarkan karena mencakup Medium dan Low. Protokol label tersebut dipertahankan sepanjang empat batch, tanpa relabel mengikuti prediksi model."
    )
    h("Fitur sensor dipisahkan dari metadata")
    p(
        "Classifier menerima fitur eye dan HRV. Nama peserta, ID sesi, difficulty, fase, judul soal, serta timestamp absolut dipakai untuk penelusuran dan pengaturan evaluasi, bukan sebagai predictor. ID sesi dapat memilih statistik normalisasi yang dihitung dari fitting, tanpa dimasukkan sebagai kelas identitas pada classifier."
    )

    # Page 3
    page("2 Evaluasi dan cara membaca hasil")
    p(
        "Keempat batch memakai evaluasi internal berbasis blok waktu dalam sesi. Orang yang sama boleh muncul pada Train, validation, dan Test, tetapi interval pendukung fitur yang bertabrakan antarbagiannya dikeluarkan. Ini bukan evaluasi leave-one-participant-out dan bukan training satu model khusus untuk setiap peserta."
    )
    p(
        "Train mempelajari parameter model. Validation memilih konfigurasi atau bentuk model akhir. Test menilai model yang telah ditetapkan. Out-of-fold (OOF) menggabungkan prediksi dari fold yang tidak ikut fitting bagi baris tersebut; ketika OOF dipakai memilih konfigurasi, nilainya tetap bagian proses pengembangan. Purge membuang window di sekitar batas agar dukungan sensor tidak bocor antarbagian."
    )
    table(
        "Perubahan rancangan evaluasi",
        ["Batch", "Blok waktu", "Purge atau guard", "Pemakaian setelah seleksi"],
        [
            ["02", "120 detik", "45 detik", "Refit Train + validation sebelum Test"],
            ["03", "120 detik", "45 detik", "Bundle validation yang sama untuk Test"],
            ["04", "240 detik", "105 detik", "Lima fold, shortlist, validation akhir"],
            [
                "05",
                "240 detik",
                "45 detik",
                "Tanpa history, satu model dikunci dari OOF",
            ],
        ],
        [0.55, 1.05, 1.4, 3.6],
    )
    p(
        "Accuracy menunjukkan proporsi prediksi benar. Macro-F1 merata-ratakan F1 ketiga kelas sehingga satu kelas tidak tenggelam oleh kelas lain. Recall menunjukkan bagian label tertentu yang berhasil dikenali. Skor keseluruhan harus dibaca bersama recall, sebab model dapat memperoleh accuracy yang tampak cukup sambil tidak pernah mengenali salah satu kelas."
    )
    figure(
        WORK / "finalists.png",
        "Accuracy Test tiap finalist per-window. Nomor Exp mengikuti folder saat ini. Dataset dan endpoint berbeda antarbatch, sehingga grafik bukan perbandingan algoritma pada kondisi identik.",
        width=6.1,
    )

    # Page 4
    page("3 Batch 02 pencarian model awal")
    p(
        "Batch 02 memakai dataset setelah integrasi dan penggantian selektif data Mon serta Alif, sebelum kebijakan kualitas yang kemudian dipakai Batch 04. Split menggunakan blok 120 detik, target pembagian mendekati 70/15/15, dan purge 45 detik. Test berisi 464, 423, dan 338 window untuk panjang 15, 25, dan 45 detik."
    )
    h("Exp01 model tabular dan kombinasi sensor")
    p(
        "Penyaringan membandingkan Logistic Regression, SVM RBF, KNN, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, dan CatBoost pada eye-only, HRV-only, serta fusion. Dua keluarga teratas tiap window mendapat tuning lanjutan. Kandidat juga mencakup log HRV, rasio baseline, mutual information, PCA, interaksi fitur, class weight, random oversampling, SMOTE, hierarchy Low/Elevated, soft voting, dan late fusion."
    )
    h("Exp02 model neural")
    p(
        "MLP, residual MLP, dan Transformer tabular kecil dibandingkan pada eye-only dan fusion. Dua preset per arsitektur mengubah ukuran, dropout, learning rate, dan weighted loss. Pengulangan seed serta voting arsitektur menguji kestabilan kombinasi. Training maksimal 100 epoch memakai patience 12 dan checkpoint accuracy validation terbaik."
    )
    p(
        "Exp01 mencatat 210 kandidat validation dan exp02 54, atau 264 catatan kandidat secara keseluruhan. Ini bukan 264 algoritma berbeda. Enam finalist per-window menjalani refit Train + validation sebelum Test."
    )
    result_table(2)
    p(
        "MLP fusion 15 detik menghasilkan Test tertinggi: 220 benar dari 464 window, atau 47,41%. Model utama kedua notebook justru memakai 45 detik karena dipilih melalui validation. Late fusion tabular memperoleh 35,21%, sedangkan voting neural 45,86%. Pada MLP 15 detik, recall Low 39,01%, Medium 41,04%, dan High 62,67%; masalah tidak terbatas pada satu kelas minoritas."
    )

    # Page 5
    page("4 Temuan Batch 02 dan perbaikan berikutnya")
    h("Pemilihan model dan refit belum stabil")
    p(
        "Model tabular utama 45 detik turun dari accuracy validation 60,92% menjadi Test 35,21%; model neural utama turun dari 61,84% menjadi 45,86%. Diagnosis membandingkan checkpoint sebelum dan sesudah refit pada endpoint Test yang sama. Refit menurunkan accuracy keenam finalist sebesar 2,37 sampai 10,17 poin persentase. Temuan tersebut menunjukkan bahwa menambahkan validation ke fitting tidak otomatis memperbaiki generalisasi."
    )
    h("Normalisasi antarsesi membantu pada kontrol tertentu")
    p(
        "Probe Logistic Regression dengan fitur, data, dan parameter tetap membandingkan normalisasi global dengan pemusatan per sesi. Pusat sesi hanya dihitung dari Train. Accuracy Test berubah dari 35,34% menjadi 54,53% pada 15 detik, 34,99% menjadi 58,39% pada 25 detik, serta 39,35% menjadi 60,36% pada 45 detik. Ini bukti diagnostik manfaat pemusatan pada konfigurasi tersebut, bukan hasil model utama yang dipilih sebelum Test."
    )
    h("Kalibrasi SVM merusak sebagian keputusan kandidat")
    p(
        "Pada SVM fusion 25 detik, keputusan model dasar memperoleh Test 43,50%, sedangkan argmax probabilitas hasil kalibrasi hanya 25,30%. Bobot SVM dasar tidak berubah. Probe fold kalibrasi memperbaiki hasil terkalibrasi menjadi 44,68%, tetapi tidak ditetapkan sebagai solusi final karena kedekatan waktu juga harus dikendalikan. Finalist resmi tidak memakai SVM, sehingga masalah ini menjelaskan kandidat SVM, bukan seluruh kesalahan finalist."
    )
    h("Skor tinggi pada diagnosis tidak mengganti skor resmi")
    p(
        "Pemeriksaan kandidat setelah Test dibaca menemukan HistGradientBoosting 45 detik dengan accuracy 64,50%. Angka itu menunjukkan bahwa peringkat validation melewatkan kandidat yang lebih baik pada bagian Test tersebut. Karena pemeriksaan bersifat post-hoc, 64,50% tidak dipromosikan sebagai hasil akhir bebas bias seleksi. Demikian pula training 87 sampai 100% pada beberapa model tidak menunjukkan keberhasilan pada interval baru."
    )
    h("Transisi menuju Batch 03")
    p(
        "Dataset tidak berubah. Batch 03 menambahkan lima fold internal ber-purge, membandingkan normalisasi global dan per sesi, serta menguji kelompok fitur nonspatial dan relatif. SVM menggunakan keputusan langsung pada jalur yang sesuai. Pemantau epoch dipisahkan dari fold penilaian neural, dan model untuk Test menjadi bundle yang sama dengan validation akhir tanpa refit Train + validation. Perubahan ini menargetkan masalah yang ditemukan, bukan sekadar menambah algoritma."
    )

    # Page 6
    page("5 Batch 03 normalisasi dan seleksi model")
    p(
        "Batch 03 mempertahankan tiga CSV historis yang sama dengan Batch 02 dan endpoint Test yang sama per-window. Bagian Train serta validation akhir disusun kembali. Lima fold di dalam Train menghasilkan OOF untuk tuning, kemudian validation akhir memilih bentuk model. Identitas peserta tetap tidak menjadi predictor."
    )
    h("Perlakuan yang diuji")
    p(
        "Exp01 membandingkan Logistic Regression, SVM, HistGradientBoosting, XGBoost, dan CatBoost melalui normalisasi global, session-centered, dan session-standardized. Kelompok input mencakup fusion 23 fitur, nonspatial 11 fitur, fitur relatif, class weight, serta kontrol eye-only/HRV-only. Exp02 membandingkan MLP, residual MLP, dan Transformer dengan normalisasi serupa, paket regularisasi, weighted loss, serta ensemble seed."
    )
    p(
        "Sebanyak 151 konfigurasi tabular dan 93 konfigurasi neural menghasilkan 1.220 hasil fold, kemudian 60 bentuk model dibandingkan pada validation akhir. Batch ini tidak menggunakan ROS atau SMOTE. Ensemble neural merata-ratakan seed 42, 43, dan 44 tanpa memilih seed tunggal dari Test."
    )
    result_table(3)
    p(
        "HistGradientBoosting 15 detik mencatat hasil tertinggi 45,91%, atau 213 benar dari 464 window. Model utama exp01 adalah SVM 25 detik dengan Test 42,08%. Model utama exp02 adalah residual MLP 45 detik dengan Test 44,08%, setelah accuracy Train 97,65% dan validation 66,49%."
    )
    h("Kesimpulan dan kebutuhan pembaruan data")
    p(
        "Perbaikan normalisasi, pemantauan, dan seleksi tidak menghasilkan kenaikan Test yang konsisten. Gap validation-Test tetap sekitar 15 sampai 26 poin persentase. Fusion lama masih mencakup fitur gaze dari sumber yang kelayakan spasialnya perlu dipisahkan dari kualitas core. Sebelum Batch 04, fokus beralih ke kelayakan masing-masing kelompok fitur dan aturan admission dataset, tanpa mengganti label untuk mengejar skor."
    )

    # Page 7
    page("6 Pembaruan dataset sebelum Batch 04")
    p(
        "Pembaruan membedakan kelayakan fitur core dari kelayakan gaze spasial. Gagal validasi pemetaan gaze tidak otomatis berarti EAR, blink, iris geometris, dan HRV pada rekaman yang sama tidak dapat dipakai. Keputusan ini menghindari pembuangan seluruh modalitas hanya karena satu kelompok fitur tidak memenuhi syarat."
    )
    h("Validasi gaze dan nilai yang tidak dapat dipercaya")
    p(
        "Validasi terakhir diperiksa terhadap konfigurasi serta prediksi lima titik uji. Penerimaan manual tidak mengubah kegagalan menjadi kelulusan. Dalam CSV terkurasi, statistik koordinat gaze dan rasio eye-state menjadi missing jika sumber atau window tidak layak. Nilai tidak diganti dengan nol, koordinat tidak ditebak, dan nilai asal tetap tersedia pada all_data. Profil core dan core-head tidak menggunakan fitur spasial tersebut."
    )
    h("Admission berdasarkan fitur yang benar-benar digunakan")
    p(
        "Core memerlukan coverage eye, face/EAR, iris, dan RR yang memadai serta nilai fitur yang finite. Ketersediaan rasio baseline tidak diwajibkan untuk core. Pembatasan kuota penggantian historis dan review berdasarkan nama tidak lagi mengunci jumlah yang diterima. Semua window yang memenuhi aturan inti dipertahankan. Sumber akuisisi kanonis tetap sama; jumlah baris bertambah karena admission yang sesuai kelompok fitur, bukan karena penambahan peserta baru pada tahap ini."
    )
    table(
        "Kelompok input setelah pembaruan",
        ["Profil", "Jumlah fitur", "Isi utama"],
        [
            [
                "Core",
                "10",
                "RMSSD, SDNN, blink rate, empat statistik EAR, tiga statistik iris",
            ],
            [
                "Core-head",
                "14",
                "Core ditambah mean dan standar deviasi pitch serta yaw",
            ],
            [
                "Validated fusion",
                "22",
                "Core-head ditambah statistik gaze dan rasio eye-state yang memenuhi validasi",
            ],
        ],
        [1.4, 0.9, 4.3],
    )
    h("Kontrol RR dan integritas waktu")
    p(
        "Window terkurasi yang mengandung RR diterima dengan deviasi lebih dari 250 ms dari median referensi kausal dikeluarkan. Nilai RR dan fitur tidak diganti atau diinterpolasi. Aturan ini mengeluarkan 27, 40, dan 64 window core layak pada panjang 15, 25, dan 45 detik. Sumber ambigu timestamp Alif tetap ditangani dengan eksklusi window yang menyentuh batas terkait, tanpa mengubah waktu mentah."
    )
    p(
        "Setelah pembaruan dan kontrol RR, total good_acceptable menjadi 5.253, 4.608, dan 3.509 window. Tiga CSV inilah yang digunakan Batch 04 dan 05. Pemeriksaan tidak menemukan missing/infinity, ID window duplikat, atau fitur konstan pada core. Kelulusan teknis ini tidak dengan sendirinya membuktikan bahwa ketiga label akan terpisah secara fisiologis."
    )

    # Page 8
    page("7 Batch 04 perluasan representasi dan model")
    p(
        "Batch 04 memakai dataset yang telah diperbarui, blok waktu 240 detik, dan purge konservatif 105 detik untuk mengakomodasi history. Test terdiri dari 768, 683, dan 529 window. Lima fold internal serta pengulangan shortlist mendahului validation akhir. Bundle yang dipilih digunakan langsung untuk Test tanpa refit setelah seleksi."
    )
    p(
        "Exp01 menguji model tabular, fitur turunan, MI/PCA, ablation sensor, class weight, ROS, SMOTE, ordinal/hierarchy, dan ensemble. Exp02 menguji MLP, residual MLP, TabM, Transformer, regularisasi, preprocessing, serta ensemble seed/keluarga. Exp03 menguji GRU/TCN dengan history 15/30/60 detik dan pembanding MLP/HistGradientBoosting. Penamaan ini mengikuti folder eksperimen yang sekarang."
    )
    p(
        "Leaderboard CV mencatat 512 konfigurasi exp01, 287 exp02, dan 157 exp03 pada tiga window, termasuk kontrol. Satu rancangan tambahan yang tidak mempunyai hasil lengkap tidak digunakan untuk menyimpulkan performa."
    )
    result_table(4)
    p(
        "TabM core-head 45 detik menghasilkan Test tertinggi 44,80%, atau 237 benar dari 529 window. Dummy yang selalu memprediksi Low memperoleh 40,64%, sehingga selisihnya hanya 22 prediksi benar. Recall TabM adalah Low 66,98%, Medium 27,87%, dan High 30,73%. Neural temporal belum memberi keuntungan yang konsisten dibanding model tabular."
    )

    # Page 9
    page("8 Diagnosis Batch 04 dan transisi menuju Batch 05")
    h("Accuracy menutupi kegagalan mengenali kelas tertentu")
    p(
        "GRU utama 45 detik memperoleh Test 40,08%, tetapi recall High nol dan recall Medium 16,39%. TabM 15 detik juga tidak memprediksi Medium. Aturan seleksi accuracy-first masih dapat menerima model yang kehilangan satu kelas. Sebagian kandidat mencapai training lebih dari 90%, sedangkan Test berada pada kisaran 30 sampai 40%."
    )
    h("Checkpoint awal berkaitan dengan masalah pemantauan")
    p(
        "Sebanyak 16 dari 24 recipe finalist deep learning tabular dan 12 dari 15 recipe temporal memakai durasi akhir satu epoch. Ini bukan bukti bahwa seluruh training CV hanya berjalan satu epoch. Diagnosis menemukan perbedaan komposisi kelas antara fitting dan monitor serta pemburukan loss monitor ketika fitting berlanjut. Cakupan sesi-kelas juga belum terjaga pada seluruh bagian evaluasi."
    )
    h("Window berdekatan memberikan skor yang sangat berbeda")
    table(
        "Kontrol 1-nearest-neighbor di dalam Train",
        [
            "Window",
            "OOF acak",
            "OOF blok purged",
            "Baris acak yang overlap dengan fitting",
        ],
        [
            ["15s", "84,17%", "41,40%", "97,62%"],
            ["25s", "93,16%", "43,48%", "98,73%"],
            ["45s", "98,39%", "49,00%", "98,68%"],
        ],
        [0.65, 1.0, 1.4, 3.55],
    )
    p(
        "Kumpulan baris dan kelas OOF sama pada kedua kontrol, tetapi assignment fold dan retensi fitting berubah karena purge. Pada skema blok tidak ada overlap dengan fitting. Angka acak 84 sampai 98% menunjukkan sensitivitas terhadap kedekatan waktu, bukan keberhasilan classifier pada interval baru. Kontrol ini tidak mengganti pembagian Test resmi."
    )
    h("Transisi menuju Batch 05")
    p(
        "Dataset tidak berubah. Batch 05 membatasi training utama pada window 25 detik, core10, dan history nol agar beberapa mekanisme dapat diuji secara terarah. Development memakai blok 240 detik dan guard 45 detik, dengan Test referensi tetap dipisahkan. Monitor dipilih dengan syarat cakupan tiga kelas, keseimbangan terhadap fitting, dan pemisahan interval."
    )
    p(
        "Pemilihan kandidat beralih ke Macro-F1 dengan recall positif pada ketiga kelas. Kontrol normalisasi dan fitting memakai fold yang sama. Neural menyelesaikan durasi tetap sehingga checkpoint loss, Macro-F1, dan epoch terakhir dapat dibandingkan dari lintasan yang sama. Tujuannya memeriksa mekanisme pipeline, bukan kembali memperluas pencarian arsitektur tanpa batas."
    )

    # Page 10
    page("9 Batch 05 kontrol pipeline yang terarah")
    h("Tiga tahap pada satu notebook")
    p(
        "Tahap 1 memeriksa identitas tiga CSV serta membekukan development dan monitor. Training utama 25 detik menggunakan 2.724 baris: 924 Low, 875 Medium, dan 925 High. Validation berisi 870 baris dan Test referensi 683; 331 baris terkena purge. Angka kelas fitting sudah berdekatan sebelum kontrol balancing tambahan."
    )
    p(
        "Tahap 2 menjalankan Logistic Regression, Extra Trees, HistGradientBoosting, dan Dummy pada normalisasi global, session-centered, serta session-robust. Lima fold yang sama, kontrol pembagian lama/baru, dan learning curve menghasilkan 71 fitting penelitian. Tidak ada search besar, SMOTE, atau fitur history pada tahap ini."
    )
    p(
        "Tahap 3 membandingkan satu MLP 64/32 dengan tiga normalisasi, tiga seed, dan tiga aturan checkpoint. Seluruh lintasan menyelesaikan 120 epoch. Kontrol Extra Trees menyamakan pool fitting neural atau mempertahankan pool penuh. Sebanyak 48 lintasan MLP dan 24 fitting tabular selesai, menghasilkan 13 kandidat OOF."
    )
    table(
        "Perbandingan kandidat penting pada OOF Train",
        ["Kandidat", "Accuracy OOF", "Macro-F1 OOF"],
        [
            ["Extra Trees penuh global", "34,03%", "34,11%"],
            ["Extra Trees penuh session-robust", "47,03%", "46,86%"],
            ["MLP session-robust checkpoint Macro-F1", "45,26%", "45,08%"],
            ["Extra Trees matched session-robust", "42,55%", "42,28%"],
        ],
        [3.8, 1.4, 1.4],
    )
    p(
        "Extra Trees penuh session-robust tetap menang berdasarkan OOF sebelum hasil validation dibaca. MLP belum melampauinya. Statistik normalisasi berasal dari fitting; validation/Test tidak digunakan untuk menghitung acuan sesi."
    )
    table(
        "Hasil model Extra Trees terkunci",
        ["Bagian", "Accuracy", "Macro-F1", "Jumlah baris"],
        [
            ["OOF Train", "47,03%", "46,86%", "2.724"],
            ["Validation", "44,83%", "43,90%", "870"],
            [
                "Test referensi",
                percent(TEST5["accuracy"]),
                percent(TEST5["macro_f1"]),
                "683",
            ],
        ],
        [2.0, 1.4, 1.4, 1.8],
    )
    p(
        "Pada Test, recall Low 33,61%, Medium 32,70%, dan High 47,01%. Model benar pada 259 window. Test ini sudah pernah dipakai dalam rangkaian analisis sebelumnya, sehingga hasil disebut referensi eksploratif, bukan pengujian independen baru."
    )

    # Page 11
    page("10 Audit penutup Batch 05")
    h("Performa tugas dan ketersediaan penilaian peserta")
    p(
        "Audit menelusuri 23 rekaman kanonis dari 22 orang yang menyumbang data pada gabungan tiga CSV. Window 25 detik sendiri mencakup 22 rekaman dari 21 orang. Sebanyak 650 attempt dapat ditautkan ke kunjungan soal terakhir pada episode set yang sama. Ketepatan attempt pertama adalah easy 58,74%, medium 30,08%, dan hard 3,70%. Ada bukti perilaku bahwa tingkat keberhasilan tugas berbeda, tetapi ketepatan jawaban bukan pengukuran langsung beban mental."
    )
    p(
        "Ledger menyimpan 167 event mulai rating dan 166 akhir rating, tanpa angka jawaban rating peserta pada sumber kanonis yang diperiksa. Skor cognitive load heuristik dari laporan sensor tidak dipakai sebagai bukti independen. Hal ini tidak menyatakan bahwa tim tidak mempunyai kuesioner di arsip lain."
    )
    h("Penelusuran sampai sensor mentah")
    p(
        "Dari prediksi Test yang sudah tersimpan dipilih satu window benar dan satu salah per kelas, mengutamakan confidence tertinggi dan blok berbeda. Rumus agregasi eye/HRV dihitung terpisah dari agregator dataset, dengan protokol penerimaan RR kausal yang sama. Seluruh 102 pemeriksaan angka cocok, dan keenam target sesuai ledger. Kasus salah prediksi pada contoh tersebut tidak dijelaskan oleh salah hitung fitur yang diperiksa."
    )
    h("Kontrol satu fitur iris yang berulang")
    p(
        "Pasangan mean iris ratio dan mean delta menunjukkan hubungan affine pada 86 dari 106 sel sesi-fold fitting yang layak, sebelum maupun sesudah normalisasi. Kontrol menghapus iris_size_delta_mean saja dan menjalankan sepuluh fitting Train-only pada fold yang sama, tanpa evaluasi validation atau Test baru."
    )
    table(
        "Pengaruh penghapusan fitur iris pada Macro-F1 OOF",
        ["Normalisasi", "Core10", "Core9", "Selisih"],
        [
            [
                "Global",
                percent(CORE9["global"]["baseline_macro_f1"]),
                percent(CORE9["global"]["core9_macro_f1"]),
                "+0,19 poin",
            ],
            [
                "Session-robust",
                percent(CORE9["session_robust"]["baseline_macro_f1"]),
                percent(CORE9["session_robust"]["core9_macro_f1"]),
                "-0,57 poin",
            ],
        ],
        [2.0, 1.4, 1.4, 1.8],
    )
    p(
        "Redundansi terbukti, tetapi penghapusannya tidak memberi perbaikan stabil. Pemeriksaan hash 1.495 file memastikan dataset, notebook, model lock, sumber terkait, dan hasil resmi tidak berubah. Seluruh 43 uji otomatis lolos. Audit ditutup tanpa mengganti pemenang atau menjalankan batch pencarian baru."
    )

    # Page 12
    page("11 Diagnosis gabungan empat batch")
    h("Ketidakseimbangan bukan penjelasan tunggal")
    p(
        "Pada dataset 25 detik terbaru, kelas berisi 1.549 Low, 1.487 Medium, dan 1.572 High. Train Batch 05 juga hampir seimbang, tetapi recall Test ketiganya tetap rendah. Class weight, ROS, dan SMOTE telah dicoba pada batch tertentu tanpa menghasilkan perbaikan akhir yang memadai. Menambah jumlah baris Low saja tidak menjelaskan kesalahan yang tersebar pada seluruh kelas."
    )
    h("Perbedaan dasar antarsesi dan pola kelas belum stabil")
    p(
        "Pada median blok 25 detik, asosiasi marginal RMSSD terhadap sesi adalah eta-squared 83,66%, sedangkan terhadap kelas 0,38%. Untuk mean iris ratio nilainya 92,23% dan 0,83%. Ini ukuran deskriptif terpisah, bukan kontribusi kausal yang dijumlahkan atau batas accuracy model. Dalam 16 sesi dengan Low dan High, arah perubahan iris ratio High terhadap Low meningkat pada tujuh sesi dan menurun pada sembilan. Menghapus kolom identitas tidak menghapus variasi yang tetap ada pada sensor."
    )
    h("Kapasitas model sudah cukup untuk mempelajari fitting")
    p(
        "Pada learning curve Extra Trees Batch 05, rata-rata Macro-F1 fitting pool penuh 95,08%, sedangkan held-out fold 47,26%. Hasil held-out meningkat ketika blok fitting bertambah, tetapi gap tetap besar. Neural juga menurunkan loss fitting sambil memburukkan loss monitor. Pola terukur ini menjelaskan mengapa menambah epoch atau arsitektur tidak otomatis menaikkan hasil pada interval lain."
    )
    figure(
        WORK / "learning_gap.png",
        "Rata-rata Macro-F1 lima fold pada learning curve Batch 05. Persentase di sumbu horizontal adalah fraksi blok fitting, bukan proporsi Test. Skor ini tidak boleh diekstrapolasi menjadi jaminan jumlah peserta untuk mencapai 70%.",
    )
    p(
        "Normalisasi membantu sebagian kontrol dan beberapa kesalahan pipeline sudah dikoreksi. Namun, manfaat OOF belum bertahan menjadi pengenalan kelas yang memadai pada validation/Test. Temuan tidak mendukung kesimpulan bahwa seluruh kegagalan berasal dari satu bug yang belum diperbaiki."
    )

    # Page 13
    page("12 Kesimpulan dan keputusan penelitian")
    h("Apa yang telah diselesaikan")
    p(
        "Rangkaian ini telah mencoba keluarga model tabular, neural tabular, Transformer, dan neural temporal; pemilihan serta turunan fitur; sensor tunggal dan fusion; penanganan imbalance; normalisasi per sesi; ensemble; serta pembenahan seleksi, refit, monitor, dan checkpoint. Dataset juga telah diperbarui berdasarkan kelayakan kelompok fitur dan integritas RR/waktu. Audit terakhir memberi bukti bahwa perhitungan teknis yang diperiksa konsisten, tanpa menghasilkan peningkatan skor yang diharapkan."
    )
    h("Apa yang belum dapat disimpulkan")
    p(
        "Kita tidak dapat menyatakan seluruh sensor rusak atau seluruh label salah. Label berasal dari protokol difficulty dan fase, bukan pengukuran beban mental independen untuk setiap window. Low juga mencakup beberapa konteks: pada 25 detik, 470 window active saja, 936 campuran active/nonactive, dan 143 nonactive saja. Protokol tetap dipertahankan, tetapi homogenitas respons fisiologis di dalam satu label tidak otomatis mengikuti keputusan tersebut."
    )
    p(
        "Fitur iris_size_estimate merupakan rasio geometri landmark iris terhadap lebar mata, bukan diameter pupil. Cleaning tidak menciptakan informasi pupil dilation yang belum direkam. Hasil audit enam kasus juga tidak sama dengan validasi fisiologis seluruh sensor atau seluruh window."
    )
    h("Keputusan yang perlu dibahas tim")
    p(
        "Pertama, laporkan hasil resmi beserta jenis evaluasinya; jangan mengganti skor dengan hasil training, pembagian acak yang overlap, atau pemilihan ulang berdasarkan Test. Kedua, tinjau apakah penilaian mental effort atau bukti pengalaman peserta tersedia di arsip terpisah dan dapat ditautkan ke tugas. Rating kesulitan dan skor sensor heuristik perlu dibedakan dari instrumen beban mental independen."
    )
    p(
        "Ketiga, bila tim melanjutkan pengumpulan atau mengubah protokol, tentukan kebutuhan pengukuran dan variasi tugas yang ingin diperkuat, bukan hanya menambah window agar jumlahnya seimbang. Urutan tugas, mutu sinyal sepanjang pengerjaan, dan hubungan label dengan respons sensor perlu dapat diperiksa. Langkah tersebut memerlukan keputusan penelitian tersendiri, bukan relabel otomatis dari prediksi model."
    )
    p(
        "Empat batch ini belum memenuhi sasaran model. Bukti sekarang tidak membuktikan bahwa 70% mustahil, tetapi belum memberi dasar untuk menjanjikan pencapaiannya melalui pencarian model besar berikutnya pada dataset yang sama. Batch 05 telah ditutup untuk pelaporan; tidak ada training tambahan yang wajib dijalankan untuk menyelesaikan auditnya."
    )
    h("Rujukan hasil dan bukti")
    p(
        "[1] Dataset fusion v2: QUALITY_REPORT.md, manifest.json, dan history/pre_feature_aware_manifest.json pada analysis_sessions/dataset_fusion/output/v2_session_phase.\n[2] Batch 02: SUMMARY.md, DIAGNOSIS.md, dan output exp01/exp02.\n[3] Batch 03: SUMMARY.md dan output exp01/exp02.\n[4] Batch 04: SUMMARY.md, DIAGNOSIS.md, diagnostic_checks.json, dan output exp01/exp02/exp03.\n[5] Batch 05: SUMMARY.md, DIAGNOSIS.md, output tahap 2/tahap 3, serta outputs/closing_audit. Seluruh folder batch berada di models. Nomor pada laporan mengikuti struktur folder saat ini.",
        "Caption",
    )

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DOC.save(DEST)
    checks = {
        "deliverable": str(DEST.relative_to(ROOT)),
        "official_finalists": len(RESULTS) + 1,
        "completed_notebooks": 8,
        "tables": TABLE_INDEX,
        "figures": FIGURE_INDEX,
        "paragraphs": len(DOC.paragraphs),
        "source_sha256": {
            str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest()
            for s in sorted(SOURCES)
        },
    }
    (WORK / "evidence_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in checks.items() if k != "source_sha256"}, indent=2)
    )


if __name__ == "__main__":
    build()
