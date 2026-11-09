"""Exclude the optional personal revision extension from public distributions."""
import re


def public_text(relative, text):
    name = str(relative).replace("\\", "/")
    if name.endswith("index.html"):
        text = re.sub(r'^.*id="cv-revision-nav".*\n', '', text, flags=re.M)
        text = re.sub(r'^.*<section id="cv-revision".*\n', '', text, flags=re.M)
    elif name.endswith("learning.js"):
        text = re.sub(r'^  let revisionIndex=.*\n', '', text, flags=re.M)
        text = re.sub(r'^    (?:if\(page!=="cv-revision"\)|\$\("cv-revision-).*$\n', '', text, flags=re.M)
        text = text.replace('    const isRevision=name==="cv-revision";\n', '')
        text = text.replace('isRevision?"cv-revision-menu":', '').replace('isRevision?"cv-revision-toggle":', '')
        text = re.sub(r'  async function showRevision\(.*?(?=  function renderSiteMap\()', '', text, flags=re.S)
        text = re.sub(r'^    (?:try\{revisionIndex=|\$\("cv-revision-nav"\)|if\(revisionIndex.available\)).*\n', '', text, flags=re.M)
        text = text.replace(',showRevision,', ',')
    elif name.endswith("app.js"):
        text = text.replace(',"cv-revision":"Private local revision"', '')
        text = text.replace(';titles["cv-revision"]="CV claims and facts revision"', '')
        text = text.replace('||name==="cv-revision"', '').replace(',"cv-revision"', '')
        text = re.sub(r'^  if\(name==="cv-revision".*\n', '', text, flags=re.M)
    elif name.endswith("style.css"):
        text = re.sub(r'^.*(?:#cv-revision|\.private-section-link).*\n', '', text, flags=re.M)
    elif name.endswith("server.py"):
        text = re.sub(r'                if path == "/api/private-revision":.*?(?=                if )', '', text, flags=re.S)
        text = re.sub(r'                if path.startswith\("/api/private-revision/"\):.*?(?=                if )', '', text, flags=re.S)
    return text
