import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup



def getHTMLContent(link):
    
    soup = BeautifulSoup(urlopen(link), 'html.parser')
    
    return soup
             

def tableParser(content, date):
    
    data = {'Country': [], 'Quality of Life Index': [],
        'Purchasing Power Index': [],'Safety Index': [],
        'Health Care Index': [], 'Cost of Living Index': [], 
        'Property Price to Income Ratio': [],
        'Traffic Commute Time Index': [],
        'Pollution Index': [], 'Climate Index': []}
    
    def dictionaryPreparation():   
            
        for el in tablehtml:
            for j in data:
                for i in range(10):
                    
                    data[j].append(tablehtml[i])
    
                    tablehtml.pop(i)
                    
                    break
            datetable['Date'].append(date)
                    
        
    table = content.find_all('table')[1]
    rows = table.find('tbody').find_all('tr')

    tablehtml = []
    datetable = {'Date': []}
    
    for row in rows:
        cells = row.find_all('td')
        for cell in cells:
            if cell.get_text() != '':
                tablehtml.append(cell.get_text().strip())
                
    dictionaryPreparation()
    dictionaryPreparation()  
    
    data.update(datetable)
    
    return data


def createDFandCSV(dataframe, dfName):
    
    dataframe.columns = headers
    dataframe.to_csv(dfName, index = False)
    


content2021mid = getHTMLContent('https://www.numbeo.com/quality-of-life/rankings_by_country.jsp?title=2021-mid')
content2021 = getHTMLContent('https://www.numbeo.com/quality-of-life/rankings_by_country.jsp?title=2021')
content2020mid = getHTMLContent('https://www.numbeo.com/quality-of-life/rankings_by_country.jsp?title=2020-mid')
content2020 = getHTMLContent('https://www.numbeo.com/quality-of-life/rankings_by_country.jsp?title=2020')
content2019mid = getHTMLContent('https://www.numbeo.com/quality-of-life/rankings_by_country.jsp?title=2019-mid')
content2019 = getHTMLContent('https://www.numbeo.com/quality-of-life/rankings_by_country.jsp?title=2019')


headers = content2021mid.find_all('table')[1].find('thead').find_all('th')[1:]
headers = [header.get_text().strip('\n') for header in headers]
headers += ['Date']


qli2021_mid = pd.DataFrame(tableParser(content2021mid, '2021-mid'))
createDFandCSV(qli2021_mid, 'qualityOfLifeIndex2021-mid.csv')

qli2021 = pd.DataFrame(tableParser(content2021, '2021'))
createDFandCSV(qli2021, 'qualityOfLifeIndex2021.csv')

qli2020_mid = pd.DataFrame(tableParser(content2020mid, '2020-mid'))
createDFandCSV(qli2020_mid, 'qualityOfLifeIndex2020-mid.csv')

qli2020 = pd.DataFrame(tableParser(content2020, '2020'))
createDFandCSV(qli2020, 'qualityOfLifeIndex2020.csv')

qli2019_mid = pd.DataFrame(tableParser(content2019mid, '2019-mid'))
createDFandCSV(qli2019_mid, 'qualityOfLifeIndex2019-mid.csv')

qli2019 = pd.DataFrame(tableParser(content2019, '2019'))
createDFandCSV(qli2019, 'qualityOfLifeIndex2019.csv')

