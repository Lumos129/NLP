import requests
from bs4 import BeautifulSoup
import json
import time
import csv
import numpy as np


csv_file = open('movies_st.csv', 'a', newline='', encoding="utf-8")
csv_writer = csv.writer(csv_file)

err_file = open('movies_err_st.csv', 'a', newline='', encoding="utf-8")
err_writer = csv.writer(err_file)

url_old = set()
url_queue = set()

idx = 0
max_movies = 0


# 电影类定义
class Movie(object):
    def __init__(self):
        self.url = None
        self.title = ''
        self.date = ''
        self.type = ''
        self.story = ''

    def __repr__(self):
        str_title = "Title: " + self.title
        str_type = "Type:  " + self.type
        str_date = "Date:  " + self.date
        str_story = "Story: " + self.story
        return '\n'.join([str_title, str_type, str_date, str_story])

    def csv_data(self):
        return [self.title, self.type, self.date, self.story]

    def csv_header(self):
        return ['Title', 'Type', 'Date', 'Story']

    def csv_data_write(self, csv_data_writer):
        csv_data_writer.writerow(self.csv_data())

    def csv_header_write(self, csv_file_handle):

        header_writer = csv.DictWriter(csv_file_handle, fieldnames=self.csv_header())
        header_writer.writeheader()


def clean_type_str(s):

    remap = {ord('\t'): None,
             ord('\n'): None,
             ord('\r'): None,
             ord(' '): None
             }
    return s.translate(remap)


def clean_story_str(s):

    remap = {ord('\t'): None,  # del \t
             ord('\n'): None,
             ord('\r'): None  # Deleted
             }
    return s.translate(remap).strip()  # 删除头尾之空格


def clean_str(s):

    remap = {ord('\t'): None,  # del \t
             ord('\n'): None,
             ord('\r'): None
             }
    return s.translate(remap).strip()  # 删除头尾之空格


def is_internal_link(href, key_words='movies.yahoo.com.tw'):
    if href.startswith('http') and key_words in href:
        return True
    return False


def is_yahoo_com_tw(href):
    if href.startswith('http') and 'movies.yahoo.com.tw' in href:
        return True
    return False


def is_yahoo_movies_info(href):
    if 'https://movies.yahoo.com.tw/movieinfo_main/' in href:
        return True
    return False


def is_old(url):
    if url in url_old:
        return True
    else:
        return False



def get_href(soup, homepage=''):
    import sys
    #print(sys._getframe().f_code.co_name)

    anchors = soup.find_all('a', href=True)
    for anchor in anchors:
        href_content = anchor['href']
        if is_yahoo_com_tw(href_content):  # 是否是本网站内链接
            if not is_old(href_content):
                url_queue.add(href_content)  # append

    return True


def get_movies(soup, url):
    import sys
    #print(sys._getframe().f_code.co_name)

    global idx

    if not is_yahoo_movies_info(url):
        return False

    movie = Movie()


    movie_intro_info_r = soup.find_all('div', attrs={'class': 'movie_intro_info_r'})
    movie_intro_info = movie_intro_info_r[0]

    try:
        # title
        title = movie_intro_info.h1
        # print(title)
        movie.title = title.contents[0]

        # type
        movie_type = movie_intro_info.a
        # print(movie_type)
        movie.type = clean_str(movie_type.contents[0])

        # date(获取多个同级span)
        # movie_time = movie_intro_info_r[0].span
        movie_spans = movie_intro_info.find_all('span')
        # print(movie_spans)
        movie_date = movie_spans[0]
        # print(movie_time.text)
        movie.date = movie_date.text[-10:]

        # story
        story = soup.find('span', attrs={'id': 'story'})
        # print(story.text)
        movie.story = clean_str(story.text)

        print(movie)
        # csv_writer.writerow(movie.csv_data())
        movie.csv_data_write(csv_writer)
        idx += 1

    except Exception as e:
        err_writer.writerow([url])
        print(e)


def parse(url, homepage):
    import sys
    print(sys._getframe().f_code.co_name)
    print(url)

    # url请求, 依赖网络与服务器, 耗时最长
    try:
        response = requests.get(url)  # timeout=？？？
        # print('requests.get is done')
        print('\033[0;32m', end='')
        print('requests.get is done', end='')
        print('\033[0m')
    except Exception as e:
        # print('requests.get is error')
        print('\033[0;31m', end='')
        print('requests.get is error', end='')
        print('\033[0m')
        print('requests.get is error', e)
        return False

    # 格式：\033[显示方式;前景色;背景色m

    soup = BeautifulSoup(response.text, 'lxml')

    # 尝试解析href, on CPU
    get_href(soup, homepage)

    # 尝试解析movies, on CPU
    get_movies(soup, url)
    # get_movies_with_goose(soup, url)

    return True


def spider(homepage):
    url_queue.add(homepage)  # homepage最先加入队列

    while url_queue:
        if idx > max_movies:
            break

        url = url_queue.pop()

        if not is_old(url):  # 查重
            url_old.add(url)
            if parse(url=url, homepage=homepage):
                time.sleep(1)


# 单线程spider
if __name__ == '__main__':
    from urllib.parse import urlparse

    # 网站主页list
    homepages = [
        "https://movies.yahoo.com.tw/movieinfo_main/LIP-X-LIP%E4%BA%AB%E5%8F%97%E9%80%99%E4%B8%96%E7%95%8C%E7%9A%84%E6%96%B9%E6%B3%95-lip-x-lip-film-x-live-11364"]

    max_movies = 50

    Movie().csv_header_write(csv_file)

    for hp in homepages:
        spider(homepage=hp)

    err_file.close()
    csv_file.close()

    exit(0)