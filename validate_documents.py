"""Check article/figure structure and reported outcomes against retained data."""
from pathlib import Path
from zipfile import ZipFile
import json,re
from lxml import etree
from PIL import Image
import numpy as np
import pandas as pd
from docx import Document
ROOT=Path(__file__).resolve().parents[1]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}

def run():
    paper=json.loads((ROOT/'manuscript_content.json').read_text());results=json.loads((ROOT/'data/results.json').read_text())
    abstract=next(b['text'] for b in paper['blocks'] if b['kind']=='abstract')
    assert 180<=len(abstract.split())<=220
    keywords=next(b['text'] for b in paper['blocks'] if b['kind']=='keywords').split('; ')
    assert 3<=len(keywords)<=6 and keywords==sorted(keywords) and all(k==k.lower() for k in keywords)
    assert len(paper['short_title'])<75
    text='\n'.join(b.get('text','') for b in paper['blocks'] if b['kind']!='reference')
    used={int(i) for match in re.findall(r'\[(\d+(?:,\d+)*)\]',text) for i in match.split(',')}
    assert used==set(range(1,26))
    assert [b['number'] for b in paper['blocks'] if b['kind']=='equation']==list(range(1,21))
    docs=[];info={}
    for suffix in ['', '_Highlighted']:
        path=ROOT/f'output/Causal_Optical_Twin_Manuscript{suffix}.docx'
        with ZipFile(path) as z:
            assert z.testzip() is None;xml=etree.fromstring(z.read('word/document.xml'))
            assert len(xml.xpath('//m:oMath',namespaces=NS))==20
            assert len(xml.xpath('//a:blip',namespaces=NS))==9
            assert len(xml.xpath('//w:tbl[not(.//m:oMath)]',namespaces=NS))==6
            assert not xml.xpath('//w:ins|//w:del',namespaces=NS)
            assert not xml.xpath('//w:instrText[contains(text(),"SEQ")]',namespaces=NS)
            highlights=xml.xpath('//w:highlight/@w:val',namespaces=NS)
            assert set(highlights)<=({'yellow'} if suffix else set())
            flat=' '.join(xml.xpath('//w:t/text()|//m:t/text()',namespaces=NS));docs.append(flat)
            rel=etree.fromstring(z.read('word/_rels/document.xml.rels'))
            links=[e.get('Target') for e in rel if e.get('Type','').endswith('hyperlink')]
            assert sum('doi.org/' in v for v in links)==24
            assert any('jmlr.org' in v for v in links)
        info[suffix or 'Clean']={'native_display_equations':20,'data_tables':6,'inline_figures':9,'research_references':25,'doi_links':24,'yellow_highlights':len(highlights)}
    assert docs[0]==docs[1]
    for fig in ROOT.glob('figures/*.png'):
        with Image.open(fig) as im:
            im.load();assert all(abs(d-600)<1 for d in im.info['dpi'])
        for ext in ['tiff','eps','pdf','svg']:assert fig.with_suffix('.'+ext).stat().st_size>0
    # Independently rederive primary reported table entries from raw prediction CSVs.
    for policy,record in results['policies'].items():
        frame=pd.read_csv(ROOT/f'data/test_{policy.lower().replace(" ","_")}.csv');z=record['nominal']
        assert len(frame)==z['n']==2000
        assert int((frame.true_class==frame.predicted_class).sum())==z['correct']
        assert int(frame.covered.sum())==z['coverage_count']
        one=frame.set_size==1;errors=(frame.true_class!=frame.predicted_class)&one
        assert int(one.sum())==z['singleton_count'] and int(errors.sum())==z['singleton_errors']
        table=next(b for b in paper['blocks'] if b['kind']=='table' and b['title'].startswith('Table 5'))
        row=next(row for row in table['rows'] if row[0]==policy)
        assert row[1]==str(z['correct']) and row[2]==f'{100*z["accuracy"]:.2f}'
    a=results['policies']['Active ML']['nominal'];fixed=results['policies']['Fixed probe']['nominal']
    assert a['accuracy']<fixed['accuracy'] and 'no demonstrated advantage' in abstract
    assert 'synthetic' in abstract.lower() and '72.50%' in abstract
    for path in ROOT.glob('output/*.docx'):
        with ZipFile(path) as z:assert z.testzip() is None
    report={'status':'passed','manuscripts':info,'clean_highlighted_content_identical':True,'abstract_words':len(abstract.split()),'alphabetized_keywords':len(keywords),'all_reported_primary_table_counts_recomputed':True,'figures_checked_600dpi':9,'visual_review':'Recorded separately after render inspection.'}
    (ROOT/'audit/DOCUMENT_VALIDATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':run()
