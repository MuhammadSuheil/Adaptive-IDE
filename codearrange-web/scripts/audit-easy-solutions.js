#!/usr/bin/env node
'use strict';
// Enumerate statement orders within the existing lexical blocks, including
// loop-body orders. Keep brace-tag order and prune out-of-scope references.
// Matching stdout identifies candidates for review, not automatic acceptance:
// fixed sample values alone cannot establish compliance with the instructions.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { normalizeOutput } = require('./validate-easy-java');
const root = path.join(__dirname, '..');
const bank = ['easy-1','easy-2','easy-3','easy-4'].flatMap(set =>
  fs.readdirSync(path.join(root,'puzzles',set)).filter(f => f.endsWith('.json')).sort().map(file => ({
    file:'codearrange-web/puzzles/'+set+'/'+file,
    puzzle:JSON.parse(fs.readFileSync(path.join(root,'puzzles',set,file),'utf8')),
  })));
const cases = JSON.parse(fs.readFileSync(path.join(root,'tests/fixtures/easy-puzzle-cases.json'),'utf8'));
const rubric = JSON.parse(fs.readFileSync(path.join(root,'tests/fixtures/easy-order-rubric.json'),'utf8'));
const strip = s => s.replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g,'');
function permutations(items) {
  if (!items.length) return [[]];
  return items.flatMap((item,index) => permutations(items.filter((_,i) => i!==index)).map(rest => [item,...rest]));
}
function parse(lines, start=0) {
  const nodes=[]; let cursor=start;
  while(cursor<lines.length && !lines[cursor].code.trim().startsWith('}')) {
    const header=lines[cursor++];
    if(!header.code.trim().endsWith('{')) { nodes.push({line:header}); continue; }
    const parts=[];
    while(true) {
      const child=parse(lines,cursor);
      cursor=child.cursor;
      const end=lines[cursor++];
      assert.ok(end,'Missing closing brace');
      parts.push({children:child.nodes,end});
      if(!end.code.includes('else')) break;
    }
    nodes.push({line:header,parts});
  }
  return {nodes,cursor};
}
function orders(nodes) {
  const alternatives=nodes.map(node => {
    if(!node.parts) return [[node.line.id]];
    let partial=[[node.line.id]];
    for(const part of node.parts) {
      const children=orders(part.children);
      partial=partial.flatMap(prefix => children.map(ids => [...prefix,...ids,part.end.id]));
    }
    return partial;
  });
  return permutations(nodes.map((_,i)=>i)).flatMap(arrangement => {
    let partial=[[]];
    for(const i of arrangement) partial=partial.flatMap(prefix => alternatives[i].map(ids => [...prefix,...ids]));
    return partial;
  });
}
function declarations(code) {
  return [...strip(code).matchAll(/\b(?:int|long|double|boolean|char|String)(?:\[\])?\s+([A-Za-z_]\w*)/g)].map(m=>m[1]);
}
function names(code, variables) {
  return [...strip(code).matchAll(/\b([A-Za-z_]\w*)\b/g)]
    .filter(m => variables.has(m[1]) && strip(code).slice(0,m.index).trimEnd().slice(-1)!=='.').map(m=>m[1]);
}
function bound(nodes, available, variables) {
  const known=new Set(available);
  for(const node of nodes) {
    const declared=declarations(node.line.code);
    if(names(node.line.code,variables).some(name=>!known.has(name)&&!declared.includes(name))) return false;
    if(node.parts) {
      const inside=new Set([...known,...declared]);
      for(const part of node.parts) {
        if(!bound(part.children,inside,variables)) return false;
        if(names(part.end.code,variables).some(name=>!known.has(name))) return false;
      }
    } else {
      for(const name of declared) {
        if(known.has(name)) return false;
        known.add(name);
      }
    }
  }
  return true;
}
const candidates=[];
for(const {file,puzzle} of bank) {
  if(puzzle.mode==='output') continue;
  const byId=new Map(puzzle.lines.map(line=>[line.id,line]));
  const canonical=puzzle.solutions[0];
  const lines=canonical.map(id=>byId.get(id));
  const variables=new Set(lines.flatMap(line=>declarations(line.code)));
  const repeated=new Map();
  for(const line of lines) {
    const key=line.code.trimStart();
    if(!repeated.has(key)) repeated.set(key,[]);
    repeated.get(key).push(line.id);
  }
  const originalTree=parse(lines.slice(2,-2)).nodes;
  for(const body of orders(originalTree)) {
    const order=[...canonical.slice(0,2),...body,...canonical.slice(-2)];
    if([...repeated.values()].some(group=>group.length>1 && order.filter(id=>group.includes(id)).join(',')!==group.join(','))) continue;
    const arranged=body.map(id=>byId.get(id));
    if(!bound(parse(arranged).nodes,new Set(),variables)) continue;
    let source=order.map(id=>byId.get(id).code).join('\n');
    if(cases[puzzle.id].probe) {
      const close=source.lastIndexOf('}',source.lastIndexOf('}')-1);
      source=source.slice(0,close)+cases[puzzle.id].probe+'\n'+source.slice(close);
    }
    const name='Candidate'+candidates.length;
    source=source.replace(/\bpublic\s+class\s+\w+/,'class '+name);
    candidates.push({file,id:puzzle.id,order,source,name,expected:normalizeOutput(cases[puzzle.id].stdout),
      listed:puzzle.solutions.some(accepted=>JSON.stringify(accepted)===JSON.stringify(order))});
  }
}
const functions=[];
for(let i=0;i<candidates.length;i+=100) {
  functions.push('static void group'+i+'(){\n'+candidates.slice(i,i+100).map((c,j)=>
    'check('+(i+j)+',()->'+c.name+'.main(new String[0]),'+JSON.stringify(c.expected)+');').join('\n')+'\n}');
}
const source=[
  'import java.io.*; import java.nio.charset.StandardCharsets;',
  'public class CandidateAudit {',
  'static void check(int id,Runnable code,String expected){',
  'PrintStream previous=System.out; ByteArrayOutputStream output=new ByteArrayOutputStream(); Throwable error=null;',
  'try{System.setOut(new PrintStream(output,true,StandardCharsets.UTF_8));code.run();}catch(Throwable e){error=e;}finally{System.setOut(previous);}',
  'String actual=output.toString(StandardCharsets.UTF_8).replace("\\r\\n","\\n").replaceAll("[ \\\\t]+(?=\\\\n|$)","");',
  'if(error==null && actual.equals(expected)) previous.println(id);',
  '}',
  ...functions,
  'public static void main(String[] args){'+candidates.filter((_,i)=>i%100===0).map((_,i)=>'group'+(i*100)+'();').join('')+'}',
  '}',
  ...candidates.map(c=>c.source),
].join('\n');
const directory=fs.mkdtempSync(path.join(os.tmpdir(),'easy-order-audit-'));
try {
  const java=path.join(directory,'CandidateAudit.java');fs.writeFileSync(java,source);
  const compiled=spawnSync('javac',['-encoding','UTF-8','-d',directory,java],{encoding:'utf8',timeout:60000,maxBuffer:2000000,windowsHide:true});
  assert.equal(compiled.status,0,compiled.stderr);
  const executed=spawnSync('java',['-cp',directory,'CandidateAudit'],{encoding:'utf8',timeout:60000,maxBuffer:2000000,windowsHide:true});
  assert.equal(executed.status,0,executed.stderr);
  const matching=executed.stdout.trim().split(/\s+/).filter(Boolean).map(Number).map(i=>candidates[i]);
  const missing=matching.filter(candidate=>!candidate.listed);
  for(const {puzzle} of bank.filter(({puzzle})=>puzzle.mode!=='output')) {
    for(const order of puzzle.solutions) {
      assert.ok(matching.some(candidate=>candidate.id===puzzle.id && JSON.stringify(candidate.order)===JSON.stringify(order)),
        'Listed order not independently verified: '+puzzle.id+'/'+order.join(','));
    }
  }
  const key = item => (item.id || item.puzzleId)+'/'+item.order.join(',');
  const reviewed = rubric.filter(item=>item.correct===false && item.sameOutput===true);
  assert.deepEqual(missing.map(key).sort(),reviewed.map(key).sort(),
    'A new same-output order requires manual instruction/behavior review; do not add it automatically.');
  const accepted=bank.reduce((total,{puzzle})=>total+puzzle.solutions.length,0);
  console.log(JSON.stringify({puzzles:bank.length,compiledCandidates:candidates.length,
    acceptedOrders:accepted,unlistedSameOutputOrders:missing.length,
    reviewedAsInstructionViolations:reviewed.length,
    unresolved:0,scope:'Statement orders within existing blocks; unchanged brace tags and distractor exclusion.',
    rejects:reviewed.map(({puzzleId,order,note})=>({puzzleId,order,note}))},null,2));
} finally {fs.rmSync(directory,{recursive:true,force:true});}
