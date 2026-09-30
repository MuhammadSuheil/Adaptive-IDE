import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import benchmark_camera


class BenchmarkTests(unittest.TestCase):
    def test_threaded_report_does_not_shadow_camera_during_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            stream = Mock()
            stream.camera_info = {}
            stream.diagnostics.return_value = {}
            stream.stopped = False
            stream.read_latest.return_value = True, object(), 1, .05
            with patch.object(benchmark_camera, 'WebcamStream') as factory, \
                 patch.object(benchmark_camera.time, 'perf_counter', side_effect=[0, 0, 0, .06, .07, .2]), \
                 patch('sys.argv', ['benchmark_camera', '--threaded', '--warmup', '0',
                                    '--seconds', '.1', '--output', str(output)]):
                factory.return_value.start.return_value = stream
                benchmark_camera.main()
            stream.stop.assert_called_once()
            self.assertEqual(json.loads(output.read_text())['frames'], 1)


if __name__ == '__main__':
    unittest.main()
