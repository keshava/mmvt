import scholarly
import requests
import bibtexparser
import os.path as op
import os
import time
import glob
import pickle
from tqdm import tqdm

# from bs4 import BeautifulSoup

# import hashlib
# import random

# _HOST = 'https://scholar.google.com'
# _SCHOLARPUB = '/scholar?oi=bibs&hl=en&cites={0}'
# _GOOGLEID = hashlib.md5(str(random.random()).encode('utf-8')).hexdigest()[:16]
# _COOKIES = {'GSP': 'ID={0}:CF=4'.format(_GOOGLEID)}
# _HEADERS = {
#     'accept-language': 'en-US,en',
#     'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/41.0.2272.76 Chrome/41.0.2272.76 Safari/537.36',
#     'accept': 'text/html,application/xhtml+xml,application/xml'
# }


# ['Constraining compartmental models using multiple voltage recordings and genetic algorithms', 'A study of computational and human strategies in revelation games', 'An agent design for repeated negotiation and information revelation with people', 'Predicting human strategic decisions using facial expressions', 'The interactive electrode localization utility: software for automatic sorting and labeling of intracranial subdural electrodes', "Instructions for users of the module'Capacity'", 'Biologically inspired load balancing mechanism in neocortical competitive learning', 'Learning to reveal information in repeated human-computer negotiation', 'Decoding hidden cognitive states from behavior and physiology using a Bayesian approach', 'Cortical Signal Suppression (CSS) for Detection of Subcortical Activity Using MEG and EEG', 'Consistent linear and non-linear responses to invasive electrical brain stimulation across individuals and primate species with implanted electrodes', 'Caudate stimulation enhances learning', 'Decoding task engagement from distributed network electrophysiology in humans', 'Estimating robustness and parameter distribution in compartmental models of neurons']

def get_author(author_name):
    # Retrieve the author's data, fill-in, and print
    search_query = scholarly.search_author(author_name)
    author = next(search_query).fill()
    return author


def get_citations_url_scholarbibs_by_publication_title(publication_title):
    bibtex_refs = []
    search_query = scholarly.search_pubs_query(publication_title)
    try:
        pub = next(search_query).fill()
        # pub = pub.fill()
        citations = list(pub.get_citedby())
        # citation = get_citatations()
        print('{} citatations for {}'.format(len(citations), pub.bib['title']))
        for citation in citations:
            bibtex_refs.append(citation.url_scholarbib)
        return bibtex_refs
    except:
        print('Can\'t find "{}"!'.format(publication_title))
        return []


# def get_citatations(pub):
#     url = _SCHOLARPUB.format(requests.utils.quote(pub.id_scholarcitedby))
#     soup = _get_soup(_HOST + url)
#     return _search_scholar_soup(soup)
#
#
#
# def _search_scholar_soup(soup):
#     """Generator that returns Publication objects from the search page"""
#     while True:
#         for row in soup.find_all('div', 'gs_or'):# {'class':'gs_r gs_or gs_scl'}):
#             yield scholarly.Publication(row, 'scholar')
#         if soup.find(class_='gs_ico gs_ico_nav_next'):
#             url = soup.find(class_='gs_ico gs_ico_nav_next').parent['href']
#             soup = _get_soup(_HOST+url)
#         else:
#             break


# def _get_soup(pagerequest):
#     """Return the BeautifulSoup for a page on scholar.google.com"""
#     html = _get_page(pagerequest)
#     html = html.replace(u'\xa0', u' ')
#     soup = BeautifulSoup(html, 'html.parser')
#     return soup


# def _get_page(pagerequest):
#     """Return the data for a page on scholar.google.com"""
#     resp = requests.Session().get(pagerequest, headers=_HEADERS, cookies=_COOKIES)
#     if resp.status_code == 200:
#         return resp.text
#     else:
#         raise Exception('Error: {0} {1}'.format(resp.status_code, resp.reason))
#

def get_citations_url_scholarbibs(publication):
    bibtex_refs = []
    publication = publication.fill()
    citations = publication.get_citedby() # get_citatations(publication) #
    print('{} citatations for {}'.format(publication.citedby, publication.bib['title']))
    for id, citation in enumerate(citations):
        bibtex_refs.append(citation.url_scholarbib)
        print(id)
    return bibtex_refs


def get_publications_url_scholarbibs(author, publications=None):
    citations_url_scholarbibs = {}
    for publication in author.publications:
        publication_title = publication.bib['title']
        if publications is not None and publication_title not in publications:
            continue
        bibtex_refs = get_citations_url_scholarbibs(publication)
        citations_url_scholarbibs[publication_title] = bibtex_refs
    return citations_url_scholarbibs


def download_publications_bibtex(citations_url_scholarbibs, fol):
    make_dir(fol)
    for publication_title, url_scholarbibs in citations_url_scholarbibs.items():
        make_dir(op.join(fol, publication_title))
        for citation_num, url_scholarbib in enumerate(url_scholarbibs):
            bibtex_fname = op.join(fol, publication_title, 'citation{}.bib'.format(citation_num + 1))
            bibtex = requests.get(url_scholarbib).content.decode()
            time.sleep(.1)
            with open(bibtex_fname, 'w') as f:
                print('Writing bibtex to {}'.format(bibtex_fname))
                f.write(bibtex)


def make_dir(fol):
    if not op.isdir(fol):
        os.makedirs(fol)


def parse_bibtex_files(fol, recursive=False):
    from collections import Counter, defaultdict
    all_authors = []
    papers = set()
    authors_papers = defaultdict(list)
    # https://bibtexparser.readthedocs.io/en/master/tutorial.html
    bib_fnames = glob.glob(op.join(op.join(fol, '**', '*.bib')), recursive=True) if recursive \
        else glob.glob(op.join(op.join(fol, '*.bib')))
    for bib_fname in bib_fnames:
        parent_fol = op.split(bib_fname)[-2]
        if recursive and parent_fol == fol:
            # Read only files in subfolders
            continue
        own_paper_name = namebase(bib_fname)
        with open(bib_fname) as bibtex_file:
            bib = bibtexparser.load(bibtex_file)
            print('{} has {} entries'.format(bib_fname, len(bib.entries)))
            for entry in bib.entries:
                paper_name = entry['title']
                # avoid duplicates
                if paper_name in papers:
                    continue
                papers.add(paper_name)
                authors = parse_authors(entry)
                for author in authors:
                    authors_papers[author].append('"{}" ({}) -> "{}"'.format(
                        paper_name, entry['year'] if 'year' in entry else '', own_paper_name))
                all_authors.extend(authors)
    all_authors = Counter(all_authors).most_common()
    print(all_authors)
    print('{} has cited you in {} publications!'.format(all_authors[0][0], all_authors[0][1]))
    # print(authors_papers[all_authors[0][0]])
    return all_authors, authors_papers


def parse_authors(bib_entry):
    return [name.strip() for name in bib_entry['author'].split('and')]


def get_bib_files(fol, recursive):
    return glob.glob(op.join(op.join(fol, '**', '*.bib')), recursive=True) if recursive \
        else glob.glob(op.join(op.join(fol, '*.bib')))

def export_bibtex(author_name, fol, recursive=False):
    from bibtexparser.bwriter import BibTexWriter
    from bibtexparser.bibdatabase import BibDatabase
    db = BibDatabase()

    papers = set()
    bib_fnames = get_bib_files(fol, recursive)
    for bib_fname in tqdm(bib_fnames):
        with open(bib_fname) as bibtex_file:
            bib = bibtexparser.load(bibtex_file)
            for entry in bib.entries:
                paper_name = entry['title']
                if paper_name in papers:
                    continue
                papers.add(paper_name)
                authors = parse_authors(entry)
                if author_name in authors:
                    db.entries.append(entry)

    author_name = author_name.replace(' ', '').replace(',', '_')
    bibtex_fname = op.join(fol, '{}.bib'.format(author_name))
    writer = BibTexWriter()
    with open(bibtex_fname, 'w') as bibfile:
        bibfile.write(writer.write(db))
    print('The bibtex file with {} papers of {} where she cited you was exported to {}'.format(
        len(db.entries), author_name, bibtex_fname))


def find_authors_papers(authors_papers, authors):
    for author_name, author_papers in authors_papers.items():
        if any([author in author_name for author in authors]):
            print(author_name)
            for paper_name in author_papers:
                print(paper_name)

def save(fname, obj):
    with open(fname, 'wb') as fp:
        pickle.dump(obj, fp)


def load(fname):
    with open(fname, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def namebase(fname):
    name_with_ext = fname.split(op.sep)[-1]
    ret = '.'.join(name_with_ext.split('.')[:-1])
    return ret if ret != '' else name_with_ext


if __name__ == '__main__':
    thesis_fol = 'C:\\Users\\peled\\Documents\\citations\\thesis'
    master_fol = 'C:\\Users\\peled\\Documents\\citations\\master'
    post_fol = 'C:\\Users\\peled\\Documents\\citations\\post'
    master_author = 'Buhry, Laure' # 'Achard, Pablo', 'De Schutter, Erik'
    thesis_author = 'Gratch, Jonathan'
    fol = post_fol
    thesis_publications = [
        'A study of computational and human strategies in revelation games',
        'An agent design for repeated negotiation and information revelation with people',
        'Predicting human strategic decisions using facial expressions',
        'Learning to reveal information in repeated human-computer negotiation'
    ]
    url_scholarbibs_fname = op.join(fol, 'url_scholarbibs.pkl')
    # 1)
    # author = get_author('Noam Peled')
    # url_scholarbibs = get_publications_url_scholarbibs(author, publications)
    # save(url_scholarbibs_fname, url_scholarbibs)
    # 2)
    # url_scholarbibs = load(url_scholarbibs_fname)
    # download_publications_bibtex(url_scholarbibs, fol)
    authors, authors_papers = parse_bibtex_files(fol)
    find_authors_papers(authors_papers, ['Achard', 'Schutter', 'Mell', 'Weerd', 'Hamilton', 'Song', 'Andersen', 'Widge'])
    # author = master_author# authors[0][0]
    # export_bibtex(author, fol)