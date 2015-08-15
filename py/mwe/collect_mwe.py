import urllib2
from BeautifulSoup import BeautifulSoup




def get_html(url):
    response = urllib2.urlopen(url)
    html = response.read()
    return html

def extract_list(html):
    soup = BeautifulSoup(html)
    divs = soup.findAll('div',attrs={'class':'mw-content-ltr'})
    div = divs[-1]
    lis = div.findAll('li')
    mwes = []
    for li in lis:
        mwes.append(li.text)
    return mwes

def get_next_page(html):
    soup = BeautifulSoup(html)
    divs = soup.findAll('div',attrs={'id':'mw-pages'})
    div = divs[0]
    ass = div.findAll('a')
    a = ass[-1]
    if a.text.startswith('next'):
        href=None
        for key,value in a.attrs:
            if key == 'href':
                href = value
                if not href.startswith('http:'):
                    href = 'http://en.wiktionary.org'+href
        return href
    else:
        return None
    


def main():
    url = 'http://en.wiktionary.org/w/index.php?title=Category:English_phrasal_verbs'
    s = 0
    f = open('mwe.txt','w')
    while url:
        print url
        html = get_html(url)
        mwes = extract_list(html)
        s += len(mwes)
        for mwe in mwes:
            f.write(mwe+'\n')
        f.flush()
        print len(mwes), s
        url = get_next_page(html)


    f.close()

main()
