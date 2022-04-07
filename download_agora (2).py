import requests
import re
import os
from bs4 import BeautifulSoup

def httpget(url):
    print('Goturl%s' % url)
    r = requests.get(url, timeout=31)
    if r.status_code == 200:
        return r.content.decode("utf-8")        
    else:
        print('Statuscode%d：%s'%(r.status_code,url))
        return
    pass

def httpDownload(url,file):
    
    try:
        r = requests.get(url, timeout=31, stream=True, allow_redirects=False)
        if r.status_code == 200:
            f = open(file, 'wb')
            for chunk in r.iter_content(chunk_size=512*1024):
                if chunk:
                    f.write(chunk)
            f.close()
        else:
            print('file%d, statuscode%s' % (file,r.status_code,url))
            return None
    except:
        print('file%s，url%s' % (file,url))
        return None
    pass

def get_ty_links():
    
    years = []
    year_links = []
    for i in range(1997,2020):
        years.append(str(i))
        year_links.append('http://agora.ex.nii.ac.jp/digital-typhoon/year/wnp/'+str(i)+'.html.en')

    tys = []
    ty_links = []
    for i in range(0,len(years)):
        html = httpget(year_links[i])
        soup = BeautifulSoup(html,"html.parser")
        row1 = soup.find_all(attrs={"class":"ROW1"})
        row0 = soup.find_all(attrs={"class":"ROW0"})
        number = len(row1)+len(row0)
        print("number",number)
        for j in range(1,10):
            tys.append(years[i]+'0'+str(j))
            ty_links.append('http://agora.ex.nii.ac.jp/digital-typhoon/summary/wnp/k/'+\
                            years[i]+'0'+str(j)+'.html.en')
        for j in range(10,number+1):
            tys.append(years[i]+str(j))
            ty_links.append('http://agora.ex.nii.ac.jp/digital-typhoon/summary/wnp/k/'+\
                            years[i]+str(j)+'.html.en')
    print("years",tys)
    print("yearlinks",ty_links)
    return tys,ty_links

def download_imgs(tys,ty_links):

    path_ = os.path.abspath('/content/drive/MyDrive/TCIE')
    root = path_ + '/Dataset/'
    if not os.path.exists(root):
        os.mkdir(root)
    
    for i in range(0,len(ty_links)):
        html = httpget(ty_links[i])
        soup = BeautifulSoup(html,"html.parser")
        a_list = soup.find_all('a')
        for a in a_list:
            if a.string.find("Image") == -1:
              print(" Not contains Image")
              continue
            image_link = 'http://agora.ex.nii.ac.jp/'+ a['href']
            html_new = httpget(image_link)
            soup_new = BeautifulSoup(html_new,"html.parser")
            tr_list = soup_new.find_all('tr') 
            boo = False
            wind = '0'
            
            text= '\n\tMaximum Wind\n      '
            for tr in tr_list:
                if (tr.string == text):
                  tr_next = tr.next_sibling.next_sibling
                  if tr_next.string[0] == '0':
                    boo = True
                    break
                  wind = str(re.findall(r'\d+',tr_next.string))
                  print("wind",wind)
            if boo: 
                continue

            pressure = '1000'
            for tr in tr_list:
                if tr.string == '\n\tCentral Pressure\n      ':
                    tr_next = tr.next_sibling.next_sibling
                    pressure = str(re.findall(r'\d+',tr_next.string))
                    print("pressure",pressure)
            
            pict_list = []
            anew_list = soup_new.find_all('a')
            for anew in anew_list:
                if anew.string == 'Magnify this':
                    st = anew['href'].replace('/0/','/1/')
                    #print("st",st)
                    pict_list.append('http://agora.ex.nii.ac.jp'+ st)
            
            try:
                s = pict_list[2]
                filename = tys[i]+'_'+s[len(s)-19:len(s)-11]+'_'+wind+'_'+pressure
                print("filename",filename)
                filename = rename(filename)
                print("filename2",filename)
                httpDownload(s,root+filename+'.jpg')

            except Exception as e:
                print(e)

                print(tys[i]),'has been downloaded.'

def rename(fname):

    new_fname = fname.replace('[','')
    new_fname = new_fname.replace(']','')
    new_fname = new_fname.replace('u','')
    new_fname = new_fname.replace('\'','')
    return new_fname
	    
if __name__ == '__main__':

    ts,links = get_ty_links()
    download_imgs(ts,links)
