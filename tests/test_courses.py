"""Structural and scope checks for the public courses and read-only lab views."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from urllib.parse import urljoin, urlsplit, unquote, parse_qs

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'site'
COURSES=('ml','dl-basics','architectures','nlp','llm')
EXPECTED={'index.html','reader.css','reader.js','course-catalog.json','release.json','.nojekyll'}|{
    'courses/'+c+s for c in COURSES for s in ('.html','.css','.js')}
LAB_ASSETS={'lab.html','style.css','api-data.json','app.js','preview.js','lab_reader.js',
    'visuals.js','readouts.js','models.js','lab_setup.js'}
LAB_PAGES={'ml-lab','ml-results','ml-compare','dl-lab','dl-results','dl-compare','train','explore','compare'}


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
    def test_only_courses_and_lab_preview_assets(self):
        actual={str(p.relative_to(SITE)) for p in SITE.rglob('*') if p.is_file()}
        self.assertEqual({p for p in actual if not p.startswith('vendor/katex/')},EXPECTED|LAB_ASSETS)
        self.assertEqual(len(list(SITE.rglob('*.html'))),7)
        self.assertTrue((SITE/'vendor/katex/LICENSE').is_file())

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
                url=urlsplit(urljoin('https://reader.invalid/study-lab/'+str(path.relative_to(SITE)),target))
                if url.netloc!='reader.invalid':continue
                self.assertTrue(url.path.startswith('/study-lab/'),(path.name,target,'escapes repository path'))
                file=SITE/unquote(url.path.removeprefix('/study-lab/'))
                if file.is_dir():file=file/'index.html'
                self.assertTrue(file.is_file(),(path.name,target))
                if url.fragment and file.suffix=='.html':
                    self.assertIn(unquote(url.fragment),pages[file].ids,(path.name,target))
                if file.name=='lab.html' and 'page' in parse_qs(url.query):
                    self.assertIn(parse_qs(url.query).get('page',[''])[0],LAB_PAGES,(path.name,target))

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
        markers=('local-nav.js','cv-revision','private-revision','writing.html','subject-python',
                 'learning_materials/','chatgpt.com/share/','PythonStages','/api/learning',
                 'thesis_hussain/','paper8/','target/cv/','projects/llm_training_lab/audit/')
        for p in SITE.rglob('*'):
            if p.is_file() and p.suffix in {'.html','.js','.css','.json'}:
                text=p.read_text()
                for marker in markers:self.assertNotIn(marker,text,(str(p.relative_to(SITE)),marker))
        for name in ('src','data','learning_materials','lab.py','pyproject.toml'):
            self.assertFalse((ROOT/name).exists(),name)

    def test_lab_catalog_and_read_only_boundary(self):
        data=json.loads((SITE/'api-data.json').read_text())
        self.assertEqual(set(data),{'/api/catalog','/api/models/catalog','/api/adaptation','/api/hardware',
                                   '/api/runs','/api/models/runs','/api/studies','/api/job'})
        for endpoint in ('/api/runs','/api/models/runs','/api/studies'):self.assertEqual(data[endpoint],[])
        self.assertIsNone(data['/api/hardware']['selected'])
        self.assertIsNone(data['/api/adaptation']['model_config'])
        self.assertFalse(data['/api/adaptation']['assets']['ready'])
        page=(SITE/'lab.html').read_text()
        self.assertLess(page.index('preview.js'),page.index('app.js'))
        self.assertLess(page.index('lab_reader.js'),page.index('app.js'))
        self.assertIn('Training, downloads and saved-run operations work only in the local application',page)
        for path in SITE.glob('*.css'):
            for target in re.findall(r'url\([\'"]?([^\)\'\"]+)',path.read_text()):
                if not target.startswith(('data:','https:','http:')):self.assertTrue((path.parent/target).is_file(),target)

    def test_static_api_adapter(self):
        if not shutil.which('node'):self.skipTest('Node is needed for adapter checks.')
        result=subprocess.run(['node',str(ROOT/'tests/test_lab_preview.cjs')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_lab_controller_navigation(self):
        if not shutil.which('node'):self.skipTest('Node is needed for controller checks.')
        result=subprocess.run(['node',str(ROOT/'tests/test_lab_ui.cjs')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_javascript_syntax(self):
        if not shutil.which('node'):self.skipTest('Node is needed for syntax checks, not for reading.')
        for p in SITE.rglob('*.js'):
            result=subprocess.run(['node','--check',str(p)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
