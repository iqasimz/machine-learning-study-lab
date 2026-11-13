"""Structural and scope checks for the six-page public course reader."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from urllib.parse import urljoin, urlsplit, unquote

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'site'
COURSES=('ml','dl-basics','architectures','nlp','llm')
EXPECTED={'index.html','reader.css','reader.js','course-catalog.json','release.json','.nojekyll'}|{
    'courses/'+c+s for c in COURSES for s in ('.html','.css','.js')}


class Page(HTMLParser):
    def __init__(self,text):
        super().__init__();self.ids=[];self.targets=[];self.math=0;self.styles=[];self.feed(text)
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if a.get('id'):self.ids.append(a['id'])
        if tag=='math':self.math+=1
        if tag in {'a','link'} and a.get('href'):self.targets.append(a['href'])
        if tag in {'script','img'} and a.get('src'):self.targets.append(a['src'])
        if tag=='link' and a.get('rel')=='stylesheet':self.styles.append(a['href'])


class CourseReleaseTests(unittest.TestCase):
    def test_exactly_six_pages_and_reading_assets(self):
        actual={str(p.relative_to(SITE)) for p in SITE.rglob('*') if p.is_file()}
        self.assertEqual(actual,EXPECTED)
        self.assertEqual(len(list(SITE.rglob('*.html'))),6)

    def test_chapters_and_version(self):
        catalog=json.loads((SITE/'course-catalog.json').read_text())
        self.assertEqual([c['id'] for c in catalog],list(COURSES))
        self.assertEqual([len(c['chapters']) for c in catalog],[13,16,8,9,10])
        version=(ROOT/'VERSION').read_text().strip()
        self.assertEqual(version,json.loads((SITE/'release.json').read_text())['version'])
        self.assertEqual(version,json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())['version'])

    def test_shared_structure_and_working_internal_links(self):
        pages={p:Page(p.read_text()) for p in SITE.rglob('*.html')}
        for path,parsed in pages.items():
            text=path.read_text()
            for tag in ('html','head','body'):
                self.assertEqual(len(re.findall('<'+tag+r'\b',text)),1,(path.name,tag))
            self.assertEqual(len(parsed.ids),len(set(parsed.ids)),path.name)
            self.assertEqual(parsed.ids.count('site-navigation'),1,path.name)
            self.assertTrue(parsed.styles[-1].endswith('reader.css'))
            for target in parsed.targets:
                if target.startswith('data:'):continue
                url=urlsplit(urljoin('https://reader.invalid/'+str(path.relative_to(SITE)),target))
                if url.netloc!='reader.invalid':continue
                file=SITE/unquote(url.path.lstrip('/'))
                if file.is_dir():file=file/'index.html'
                self.assertTrue(file.is_file(),(path.name,target))
                if url.fragment and file.suffix=='.html':
                    self.assertIn(unquote(url.fragment),pages[file].ids,(path.name,target))

    def test_five_final_summary_chapters_and_math(self):
        for course,count in zip(COURSES,[7,6,6,6,6]):
            text=(SITE/'courses'/(course+'.html')).read_text()
            parsed=Page(text)
            self.assertGreater(parsed.math,30)
            self.assertIn('summary-and-interview-questions',parsed.ids)
            summary=text.split('<article class="summary-chapter"',1)[1]
            self.assertEqual(len(re.findall(r'<h2[^>]*>Question \d',summary)),count)
            self.assertNotIn(r'\[',text)

    def test_no_extra_ui_or_local_material(self):
        markers=('lab.html','local-nav.js','api-data.json','cv-revision','private-revision',
                 'writing.html','subject-python','learning_materials/','chatgpt.com/share/')
        for p in SITE.rglob('*'):
            if p.is_file() and p.suffix in {'.html','.js','.css','.json'}:
                text=p.read_text()
                for marker in markers:self.assertNotIn(marker,text,(str(p.relative_to(SITE)),marker))
        for name in ('src','data','learning_materials','lab.py','pyproject.toml'):
            self.assertFalse((ROOT/name).exists(),name)

    def test_javascript_syntax(self):
        if not shutil.which('node'):self.skipTest('Node is needed for syntax checks, not for reading.')
        for p in SITE.rglob('*.js'):
            result=subprocess.run(['node','--check',str(p)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
