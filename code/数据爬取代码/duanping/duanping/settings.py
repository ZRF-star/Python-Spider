# Scrapy settings for duanping project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'duanping'

SPIDER_MODULES = ['duanping.spiders']
NEWSPIDER_MODULE = 'duanping.spiders'


# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'Mozilla/5.0'

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
}

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
#CONCURRENT_REQUESTS = 32

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 1
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
  'Accept-Language': 'en',
  'COOKIE': 'll="118384"; bid=GKLCw5kmGgg; __utmv=30149280.18923; douban-profile-remind=1; gr_user_id=451cdeaa-b4ae-4658-94b8-390176197a3a; _vwo_uuid_v2=D318EA409C7F88E7644A99A1028CC63C4|8adf2a3f85e3a649bf101675bc2f1f8b; douban-fav-remind=1; _ga=GA1.2.603145847.1594112487; __gads=ID=41015dff6a92438f-22fc1f2ca8c400ce:T=1604984506:RT=1604984506:S=ALNI_Ma-au_kmESc-5rl4SWD2IujxxwQ6g; ct=y; viewed="30409058_1085799_33463346"; gr_cs1_6165b581-5a1d-4c17-bc0c-3d09d34bbbe9=user_id%3A0; ap_v=0,6.0; __utmz=30149280.1609482925.21.15.utmcsr=sogou.com|utmccn=(referral)|utmcmd=referral|utmcct=/link; __utmc=30149280; __utma=30149280.603145847.1594112487.1609474914.1609482925.21; gr_cs1_7fdcd8d6-3846-4ca3-a8cd-9d1e468326d7=user_id%3A1; push_noty_num=0; push_doumail_num=0; __utmt_douban=1; gr_cs1_17771bed-bfb4-4d84-8f71-5f5d70726c31=user_id%3A0; dbcl2="189233793:bijp07v0cig"; ck=RmPg; gr_session_id_22c937bbd8ebd703f2d8e9445f7dfd03=653a30fa-61d1-44d1-954f-d841ad02f670; gr_cs1_653a30fa-61d1-44d1-954f-d841ad02f670=user_id%3A1; gr_session_id_22c937bbd8ebd703f2d8e9445f7dfd03_653a30fa-61d1-44d1-954f-d841ad02f670=true; __utmb=30149280.16.10.1609482925'
}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'duanping.middlewares.DuanpingSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    'duanping.middlewares.DuanpingDownloaderMiddleware': 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
   'duanping.pipelines.DuanpingPipeline': 300,
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
