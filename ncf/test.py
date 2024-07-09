# -*- coding: utf-8 -*-
import pandas as pd
import torch


g_dict = {'雷钧': '浙江',
'史少朋': '陕西',
'张莎莎': '贵州',
'马琳鑫': '山东',
'张龙凯': '山东',
'黄展华': '广东',
'陈洋': '山东',
'刘畅': '北京',
'丁汀汀': '四川',
'梅培根': '安徽',
'田震': '山东',
'邹嘉俊': '广东',
'张丽娅': '浙江',
'张亚单': '北京',
'崔艺迅': '河南',
'徐永定': '江苏',
'岳磊': '宁夏',
'刘洋': '海南',
'秦阔林': '湖北',
'张凯': '山东',
'岳磊': '山东',
'蓝国华': '广东',
'朱文浩': '江苏',
'金军灵': '浙江',
'金穗儿': '北京',
'刘畅': '河北',
'李泽鑫': '安徽',
'陈洋': '宁夏',
'桂洪丽': '安徽',
'范秋婧': '四川',
'张凯': '重庆',
'齐志翔': '浙江',
'陈洋': '北京',
'钟涛': '广东',
'王皓轩': '湖北',
'孙涛': '辽宁',
'刘洋': '四川',
'吴愿愿': '北京',
'高文博': '北京',
'吴春蕾': '海南',
'徐钰涵': '湖北',
'朱伟': '浙江',
'张凯': '河北',
'何师法': '广东',
'张聪': '河南',
'陶天奇': '湖南',
'高风浩': '山东',
'戴毅': '广东',
'郭欢': '江苏',
'王学成': '上海',
'朱敏': '湖南',
'刘建全': '云南',
'肖若飞': '湖北',
'张聪': '山东',
'柳玉蒙': '河南',
'池启晋': '广东',
'王芬': '浙江',
'尹永明': '吉林',
'张子洋': '湖北',
'谭彩霞': '湖北',
'陈小恒': '江苏',
'王江涛': '河南',
'张凯': '浙江',
'刘琰': '湖南',
'刘洋': '江苏',
'马月敏': '河北',
'刘洋': '海南',
'陈树立': '广东',
'罗凯夫': '广东',
'朱敏': '安徽',
'魏青臣': '河南',
'杜红江': '四川',
'商小濛': '吉林',
'王群': '江苏',
'陈洋': '河北',
'郭小玩': '广东',
'曾小霞': '浙江',
'李日春': '广东',
'肖蕾': '广东',
'仝德宝': '上海',
'杨乐石': '陕西',
'董静磊': '吉林',
'王鸿臻': '辽宁',
'刘家铭': '广东',
'钱紫依': '江苏',
'练品楠': '广东',
'姜凌宇': '辽宁',
'程丽君': '四川',
'蔡建文': '广东',
'周静驰': '广东',
'康斌': '山西',
'方碧茵': '福建',
'阳鑫': '四川',
'郭欢': '广东',
'田震': '江苏',
'刘畅': '河北',
'孙志豪': '江苏',
'刘洋': '内蒙古',
'王彦峰': '河南',
'范龙飞': '山东',
'许伟棹': '广东',
'徐志超': '江苏',
'徐波': '重庆',
'刘洋': '浙江',
'李灏': '宁夏',
'卢斌': '河北',
'刘洋': '宁夏',
'李欢': '陕西',
'郝文博': '浙江',
'王国文': '山西',
'李丽君': '四川',
'徐瑞欣': '浙江',
'杨弼海': '四川',
'魏峥': '河北',
'刘畅': '广东',}

l = ['梅培根',
'徐钰涵',
'徐钰涵',
'曹微',
'齐雪',
'王彦峰',
'王皓轩',
'王皓轩',
'谭彩霞',
'王彦峰',
'王彦峰',
'王皓轩',
'王彦峰',
'王彦峰',
'王彦峰',
'王彦峰',
'王皓轩',
'王皓轩',
'王皓轩',
'王皓轩',
'王皓轩',
'王皓轩',
'王皓轩',
'王皓轩',
'王皓轩',
'王鸿臻',
'王彦峰',
'王鸿臻',
'钟涛',
'张亚单',
'张亚单',
'张亚单',
'阳鑫',
'徐钰涵',
'徐钰涵',
'徐钰涵',
'徐瑞欣',
'吴愿愿',
'王彦峰',
'王彦峰',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'孙涛',
'孙涛',
'孙涛',
'齐雪',
'梅培根',
'梅培根',
'梅培根',
'梅培根',
'姜凌宇',
'黄展华',
'方碧茵',
'丁汀汀',
'崔艺迅',
'齐雪',
'王鸿臻',
'王彦峰',
'王彦峰',
'徐钰涵',
'阳鑫',
'王彦峰',
'徐钰涵',
'徐钰涵',
'徐钰涵',
'徐钰涵',
'徐钰涵',
'池启晋',
'徐钰涵',
'徐钰涵',
'徐钰涵',
'王彦峰',
'王彦峰',
'王彦峰',
'王彦峰',
'齐雪',
'蓝国华',
'齐雪',
'魏峥',
'齐雪',
'齐雪',
'齐雪',
'蓝国华',
'徐波',
'吴愿愿',
'何师法',]

r_dict = {}

for i in l:
    if i in r_dict.keys():
        r_dict[i] += 1
    else:
        r_dict[i] = 1

print(sorted(r_dict.items(), key=lambda x: x[1], reverse=True))

l1 = ['范龙飞',
'范龙飞',
'梅培根',
'喻佳',
'喻佳',
'徐钰涵',
'李梦',
'刘凡',
'胡剑雄',
'刘爽',
'程明丽',
'张泽旸',
'岳磊',
'陈树立',
'吕建峰',
'曹向璐',
'张子洋',
'孙志豪',
'范士媛',
'童孟欣',
'纪立权',
'黄金涛',
'徐钰涵',
'曹微',
'齐雪',
'岳磊',
'朱伟',
'岳磊',
'魏青臣',
'魏青臣',
'魏青臣',
'杨弼海',
'刘家铭',
'孙志豪',
'王彦峰',
'徐志超',
'柳玉蒙',
'朱文浩',
'王皓轩',
'雷钧',
'孙志豪',
'康斌',
'王皓轩',
'谭彩霞',
'王彦峰',
'王彦峰',
'岳磊',
'王皓轩',
'王群',
'吴春蕾',
'金军灵',
'王江涛',
'高文博',
'王彦峰',
'张子洋',
'王彦峰',
'张子洋',
'杜红江',
'卢斌',
'王彦峰',
'王彦峰',
'桂洪丽',
'李丽君',
'王皓轩',
'王学成',
'史少朋',
'张子洋',
'王皓轩',
'王皓轩',
'王皓轩',
'王皓轩',
'王皓轩',
'王皓轩',
'王皓轩',
'陈树立',
'王皓轩',
'张聪',
'田震',
'周静驰',
'田震',
'王鸿臻',
'张子洋',
'张子洋',
'王彦峰',
'王鸿臻',
'郭小玩',
'邹嘉俊',
'钟涛',
'张亚单',
'张亚单',
'张亚单',
'张凯',
'尹永明',
'阳鑫',
'徐志超',
'徐钰涵',
'徐钰涵',
'徐钰涵',
'许伟棹',
'徐瑞欣',
'肖蕾',
'吴愿愿',
'王彦峰',
'王彦峰',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'王鸿臻',
'陶天奇',
'陶天奇',
'孙涛',
'孙涛',
'孙涛',
'商小濛',
'齐雪',
'钱紫依',
'梅培根',
'梅培根',
'梅培根',
'梅培根',
'马月敏',
'马琳鑫',
'罗凯夫',
'李日春',
'刘琰',
'刘畅',
'姜凌宇',
'黄展华',
'郝文博',
'高风浩',
'范龙飞',
'方碧茵',
'丁汀汀',
'崔艺迅',
'陈洋',
'齐雪',
'王鸿臻',
'曾小霞',
'王彦峰',
'王彦峰',
'程丽君',
'刘洋',
'徐钰涵',
'杨乐石',
'曾小霞',
'阳鑫',
'王彦峰',
'雷钧',
'王国文',
'王国文',
'徐钰涵',
'徐钰涵',
'徐钰涵',
'徐钰涵',
'徐钰涵',
'池启晋',
'徐钰涵',
'徐钰涵',
'徐钰涵',
'朱敏',
'李灏',
'王彦峰',
'仝德宝',
'王彦峰',
'商小濛',
'王彦峰',
'曾小霞',
'徐永定',
'王彦峰',
'齐雪',
'曾小霞',
'郭欢',
'刘建全',
'康斌',
'蓝国华',
'齐雪',
'魏峥',
'齐雪',
'齐雪',
'齐雪',
'蓝国华',
'秦阔林',
'戴毅',
'蔡建文',
'徐波',
'齐志翔',
'金穗儿',
'孙志豪',
'陈小恒',
'李泽鑫',
'张龙凯',
'王江涛',
'吴愿愿',
'张莎莎',
'何师法',
'张丽娅',
'肖若飞',
'董静磊',
'李欢',
'练品楠',
'王芬',
'范秋婧',
'董静磊',
'马君伟',
'马君伟',
'马君伟',
'马君伟',]

l2 = ['丁汀汀',
'何师法',
'吴愿愿',
'姜凌宇',
'孙涛',
'崔艺迅',
'张亚单',
'徐波',
'徐瑞欣',
'徐钰涵',
'方碧茵',
'曹微',
'梅培根',
'池启晋',
'王彦峰',
'王皓轩',
'王鸿臻',
'蓝国华',
'谭彩霞',
'钟涛',
'阳鑫',
'魏峥',
'黄展华',
'齐雪',]


for i1 in l2:
    print(i1, '-', r_dict[i1])

# for i1 in l1:
#     if i1 in l2:
#         print(i1)