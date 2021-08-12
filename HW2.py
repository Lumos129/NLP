import requests
from bs4 import BeautifulSoup
import json
import time
import csv
import re
import numpy as np
import queue

import jieba
from collections import Counter

# from goose3 import Goose
# from goose3.text import StopWordsChinese

import threading

lock = threading.Lock()

csv_file = open('movies_mt.csv', 'a', newline='', encoding="utf-8")
csv_writer = csv.writer(csv_file)

sql_file = open('news32q.sql', 'w', encoding="utf-8")
err_file = open('movies_err.csv', 'a', newline='', encoding="utf-8")
err_writer = csv.writer(err_file)

url_old = set()
url_set = set()

idx = 0
running_thread_num = 0
max_movies = 6000



class Movie(object):
    def __init__(self):
        self.url = None
        self.title = ''
        self.date = ''
        self.type = ''
        self.story = ''

    def __repr__(self):
        str_title = "Title: " + self.title
        str_type = "Type: " + self.type
        str_time = "Date: " + self.date
        str_story = "Story: " + self.story
        return '\n'.join([str_title, str_type, str_time, str_story])

    def csv_data(self):
        return [self.title, self.type, self.date, self.story]


class SharedVar(object):
    def __init__(self):
        self.idx = 0
        self.max_movies = 10


def sort_movies_csv():
    import pandas as pd
    # 读取文件数据
    df = pd.read_csv('movies_mt.csv')

    # 按照列值排序
    data = df.sort_values(by="Type", ascending=True)

    # 把新的数据写入文件
    data.to_csv('movies_mt_sort.csv', mode='a+', index=False)


def clean_str(s):
    remap = {ord('\t'): None,  # \t替换成空格
             ord('\n'): None,
             ord('\r'): None,
             ord(' '): None  # Deleted
             }
    return s.translate(remap)


def clean_story(s):
    remap = {ord('\t'): None,  # \t替换成空格
             ord('\n'): None,
             ord('\r'): None  # Deleted
             }
    return s.translate(remap).strip()


def is_yahoo_com_tw(href):
    if 'movies.yahoo.com.tw' in href:
        return True
    return False


def is_yahoo_movies_info(href):
    if 'https://movies.yahoo.com.tw/movieinfo_main/' in href:
        return True
    return False


class SpiderThread(threading.Thread):
    def __init__(self, homepage, thread_id, global_var=None):
        threading.Thread.__init__(self)
        # super(SpiderThread, self).__init__()
        self.homepage = homepage
        self.th = thread_id
        self.g_var = global_var

        # self.g = Goose({'stopwords_class': StopWordsChinese})

    def is_old(self, url):
        if url in url_old:
            return True
        else:
            return False

    def get_movies(self, soup, url):
        import sys
        print(sys._getframe().f_code.co_name)

        global idx

        if not is_yahoo_movies_info(url):
            return False

        movie = Movie()

        # 类名为xxx而且文本内容为hahaha的div
        # for k in soup.find_all('div', class='movie_intro_list'):        #,string='更多'
        # print(k)
        movie_intro_info_r = soup.find_all('div', attrs={'class': 'movie_intro_info_r'})

        # <div class="atcTit_more"><span class="SG_more"><a href="http://blog.sina.com.cn/" target="_blank">更多&gt;&gt;</a></span></div>

        try:
            # title
            title = movie_intro_info_r[0].h1
            # print(title)
            movie.title = title.contents[0]

            # time(获取多个同级span)
            # movie_time = movie_intro_info_r[0].span
            movie_spans = movie_intro_info_r[0].find_all('span')
            # print(movie_spans)
            movie_time = movie_spans[0]
            # print(movie_time.text)
            movie.time = movie_time.text[-10:]

            # type
            movie_type = movie_intro_info_r[0].a
            # print(movie_type)
            movie.type = clean_str(movie_type.contents[0])

            # story
            story = soup.find('span', attrs={'id': 'story'})
            # print(story.text)
            movie.story = clean_story(story.text)

            print(movie)

            lock.locked()
            csv_writer.writerow(movie.csv_data())
            self.g_var.idx += 1
            lock.release()

        except Exception as e:
            # lock.locked()
            err_writer.writerow([url])
            # lock.release()
            print(e)

    # 解释href
    def get_href0(self, soup):
        import sys
        # print(sys._getframe().f_code.co_name, self.th)

        anchors = soup.find_all('a', href=True)
        # print('anchors:', len(anchors))
        for anchor in anchors:
            href_content = anchor['href']
            href_content = href_content.strip()
            if href_content.startswith('http'):
                if self.web_dict[self.homepage] in href_content:  # 是否是本网站内链接
                    lock.acquire()
                    if not self.is_old(href_content):
                        url_set.add(href_content)  # append
                    lock.release()
            else:
                if href_content.startswith('//'):
                    href_content = 'https:' + href_content
                else:
                    href_content = self.homepage + href_content

                lock.acquire()
                # add_to_url_queue(url)
                if not self.is_old(href_content):
                    url_set.add(href_content)
                lock.release()
        return True

    def get_href(self, soup, homepage=''):
        import sys
        print(sys._getframe().f_code.co_name)

        anchors = soup.find_all('a', href=True)
        for anchor in anchors:
            href_content = anchor['href']
            if is_yahoo_com_tw(href_content):  # 是否是本网站内链接
                if not self.is_old(href_content):
                    url_set.add(href_content)  # append

        return True

    # 解析url
    def parse_thread(self, url):
        import sys
        # print(sys._getframe().f_code.co_name, self.th)

        # url请求, 依赖网络与服务器, 耗时最长
        try:
            response = requests.get(url, timeout=5)
            response.encoding = 'utf-8'  # 应对中文网站编码的变化！'gb18030'
            # print('requests.get is done', self.th)
            print(url, self.th)
        except:
            # err_file.write(url+'\n')
            # print('requests.get is error', th)

            # 控制台输出红色字体
            print('\33[0;31m', end='')
            print('requests.get is error', self.th, end='')
            print('\33[0m')
            print(url, self.th)
            return False

        """def __init__(self, markup="", features=None, builder=None,
                 parse_only=None, from_encoding=None, exclude_encodings=None,
                 **kwargs):
                 """
        soup = BeautifulSoup(response.text, 'lxml')

        # 尝试解析href, on CPU
        self.get_href(soup)

        # 尝试解析新闻, on CPU
        self.get_movies(soup=soup, url=url)

        return True

    def spider_thread(self):
        """
        实际的线程入口函数
        :return:
        """
        import sys
        print(sys._getframe().f_code.co_name, self.th)

        global idx, running_thread_num, max_article

        url_set.add(self.homepage)  # homepage最先加入队列

        # 遍历url_set
        while True:
            if self.g_var.idx > self.g_var.max_movies:
                break

            # print(self.g_var.idx, self.g_var.max_movies)

            # 检查终止条件(文件Stop.txt)
            # if is_stop():
            #    print('thread', th, 'stop')
            #    return False

            lock.acquire()  # lock资源
            if len(url_set) != 0:
                url = url_set.pop()
                lock.release()  # release资源
            else:
                if running_thread_num == 0:
                    lock.release()  # release资源
                    break

                lock.release()  # release资源
                # print('continue...')
                continue

            lock.acquire()  # lock资源
            if not self.is_old(url):  # 查重
                url_old.add(url)
                running_thread_num += 1
                lock.release()  # release资源

                # 各种浪
                print(' ' * (self.g_var.idx % 50), 'queue:', len(url_set), 'old:', len(url_old), 'movies:',
                      self.g_var.idx)

                # 去解析url
                if self.parse_thread(url=url):
                    time.sleep(5)

                lock.acquire()
                running_thread_num -= 1
                lock.release()
            else:
                lock.release()  # release资源

    def run(self):
        """
        保留的线程入口函数
        :return:
        """
        self.spider_thread()


# 多线程spider
if __name__ == '__main__':
    # 网站主页list
    homepages = [
        'https://movies.yahoo.com.tw/movieinfo_main/LIP-X-LIP%E4%BA%AB%E5%8F%97%E9%80%99%E4%B8%96%E7%95%8C%E7%9A%84%E6%96%B9%E6%B3%95-lip-x-lip-film-x-live-11364/']
    # homepages = ['https://movies.yahoo.com.tw/']

    total_thread = 1000
    max_movies = 6000

    # csv文件头
    field_names = ['Title', 'Type', 'Date', 'Story']
    writer = csv.DictWriter(csv_file, fieldnames=field_names)
    writer.writeheader()

    g_var = SharedVar()

    for hp in homepages:
        try:
            # 创建多线程(100)
            ths = []
            for thn in range(total_thread):
                th = SpiderThread(hp, thn, g_var)
                th.start()  # 依次启动多线程
                ths.append(th)

            # waiting多线程终止
            for th in ths:
                th.join()
        except:
            print("Error: 无法启动线程")

        sql_file.close()
        err_file.close()
        csv_file.close()

    sort_movies_csv()

    exit(0)






