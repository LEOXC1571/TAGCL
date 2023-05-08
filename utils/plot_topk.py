# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: plot_topk.py
# @Author: Leo Xu
# @Date: 2023/3/3 14:13
# @Email: leoxc1571@163.com
# Description:

import io
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


font_title = {
    'family': 'Times New Roman',
    'color': 'black',
    'weight': 'bold',
    'size': 16
}
font_subtitle = {
    'family': 'Times New Roman',
    'color': 'black',
    'weight': 'bold',
    'size': 12
}
font_label = {
    'family': 'Times New Roman',
    'color': 'black',
    'weight': 'normal',
    'size': 12
}
colors = ['#B4C7E7', '#FF7C80', '#C5E0B4']
label_size = 12
text_size = 10

colors_map = {
    'LGCN': '#8058A5',
    'SimGCL': '#1F8DD6',
    'BPR-T': '#5EB95E',
    'TGCN': '#FAD232',
    'LFGCF': '#F37B1D',
    'TAGCL': '#DD514C'
}

data = {
    'ml': {
        'LGCN': [0.1971, 0.2312, 0.2557, 0.2788, 0.2895, 0.3079,
                 0.0685, 0.0475, 0.0398, 0.0349, 0.0314, 0.0293,
                 0.1805, 0.189, 0.1956, 0.2015, 0.2046, 0.2094,
                 0.1975, 0.205, 0.2081, 0.2101, 0.2106, 0.2116,
                 31.5538, 29.0442, 27.7554, 26.7787, 26.092, 25.4437],
        'SimGCL': [0.2225, 0.2553, 0.2719, 0.2857, 0.3023, 0.31,
                   0.0763, 0.0536, 0.0442, 0.0385, 0.0343, 0.0315,
                   0.2086, 0.2188, 0.2239, 0.2279, 0.2322, 0.2347,
                   0.2252, 0.2335, 0.2358, 0.2372, 0.2379, 0.2383,
                   19.3321, 18.7894, 18.2908, 17.8736, 17.5018, 17.1449],
        'BPR-T': [0.2177, 0.2536, 0.2695, 0.2826, 0.2957, 0.3105,
                  0.0737, 0.0522, 0.0419, 0.0365, 0.0327, 0.03,
                  0.2012, 0.2127, 0.2171, 0.2209, 0.2247, 0.2286,
                  0.2166, 0.224, 0.2263, 0.2272, 0.228, 0.2287,
                  22.3323, 22.8401, 22.8174, 22.756, 22.648, 22.4739],
        'TGCN': [0.2041, 0.2405, 0.2605, 0.2774, 0.2873, 0.3013,
                 0.0681, 0.0494, 0.0406, 0.0351, 0.0318, 0.0296,
                 0.192, 0.2042, 0.2102, 0.2147, 0.218, 0.222,
                 0.2075, 0.2163, 0.2189, 0.2202, 0.2208, 0.2214,
                 19.3013, 19.9075, 20.0087, 19.8749, 19.7659, 19.6136],
        'LFGCF': [0.2043, 0.2563, 0.2737, 0.2929, 0.3041, 0.3164,
                  0.0697, 0.0523, 0.0426, 0.0365, 0.0327, 0.0301,
                  0.1868, 0.2037, 0.209, 0.214, 0.2174, 0.2209,
                  0.2037, 0.2148, 0.2169, 0.2183, 0.219, 0.2196,
                  18.8595, 18.5821, 18.3247, 18.1001, 17.8248, 17.6247],
        'TAGCL': [0.2369, 0.2756, 0.296, 0.318, 0.3357, 0.3478,
                  0.0799, 0.0555, 0.0457, 0.0405, 0.0368, 0.0335,
                  0.2093, 0.2213, 0.2275, 0.2338, 0.239, 0.2425,
                  0.2233, 0.2313, 0.2339, 0.2356, 0.2367, 0.2371,
                  16.0166, 15.5013, 15.2262, 14.9665, 14.766, 14.6043]
    },
    'lfm': {
        'LGCN': [0.3163, 0.3926, 0.4427, 0.4742, 0.5052, 0.5293,
                 0.208, 0.1686, 0.1494, 0.135, 0.125, 0.1171,
                 0.3655, 0.3826, 0.3947, 0.4015, 0.4087, 0.4146,
                 0.4403, 0.4541, 0.4576, 0.459, 0.4598, 0.4603,
                 163.2049, 139.8825, 124.8036, 114.4561, 106.7012, 100.5721],
        'SimGCL': [0.3685, 0.4381, 0.4793, 0.5055, 0.5277, 0.5433,
                   0.2483, 0.1982, 0.172, 0.1534, 0.1406, 0.1304,
                   0.4384, 0.4531, 0.4625, 0.468, 0.4731, 0.4771,
                   0.5144, 0.5238, 0.5257, 0.5263, 0.5267, 0.5269,
                   72.2308, 61.6905, 55.6324, 51.6677, 48.7597, 46.5794],
        'BPR-T': [0.3375, 0.3969, 0.4414, 0.4759, 0.5043, 0.5256,
                  0.2261, 0.175, 0.1525, 0.1374, 0.127, 0.1183,
                  0.4143, 0.4205, 0.4291, 0.4358, 0.4421, 0.4469,
                  0.4981, 0.5083, 0.5118, 0.5132, 0.5142, 0.5146,
                  122.236, 115.4028, 108.6373, 102.8396, 98.5174, 94.9261],
        'TGCN': [0.3248, 0.3869, 0.4347, 0.4663, 0.4897, 0.5109,
                 0.2082, 0.164, 0.1448, 0.1313, 0.1217, 0.1142,
                 0.3831, 0.3949, 0.4071, 0.4149, 0.4209, 0.4266,
                 0.4576, 0.4687, 0.4716, 0.4727, 0.4734, 0.4738,
                 86.355, 85.0558, 82.8195, 80.7635, 78.4368, 76.4911],
        'LFGCF': [0.3497, 0.4253, 0.4731, 0.5057, 0.5329, 0.5517,
                  0.2273, 0.1843, 0.1618, 0.1465, 0.1361, 0.1264,
                  0.4109, 0.4286, 0.4404, 0.4482, 0.4554, 0.4601,
                  0.4877, 0.499, 0.5018, 0.5033, 0.5039, 0.5041,
                  99.5585, 91.8826, 85.1911, 80.6513, 77.1384, 74.3997],
        'TAGCL': [0.3912, 0.4593, 0.4961, 0.5199, 0.5386, 0.5534,
                  0.2639, 0.2088, 0.1802, 0.1611, 0.1469, 0.1355,
                  0.4693, 0.4823, 0.4902, 0.4949, 0.4987, 0.5021,
                  0.5444, 0.5521, 0.5535, 0.5541, 0.5544, 0.5546,
                  60.8255, 50.4132, 45.811, 42.9929, 41.0625, 39.5084]
    },
    'de': {
        'LGCN': [0.1165, 0.1973, 0.2696, 0.3337, 0.3968, 0.4541,
                 0.3739, 0.3635, 0.3585, 0.3525, 0.3506, 0.3474,
                 0.3985, 0.4032, 0.4116, 0.4213, 0.4363, 0.4527,
                 0.5644, 0.5755, 0.578, 0.5786, 0.579, 0.5791,
                 2.6247, 2.7941, 2.9566, 3.1147, 3.2979, 3.463],
        'SimGCL': [0.1139, 0.1948, 0.2689, 0.3351, 0.3963, 0.4513,
                   0.3614, 0.3596, 0.3588, 0.3554, 0.3523, 0.3473,
                   0.3818, 0.3929, 0.4052, 0.4177, 0.4323, 0.4475,
                   0.5344, 0.5492, 0.5523, 0.5529, 0.5532, 0.5533,
                   5.6307, 5.1149, 4.8356, 4.665, 4.5456, 4.5152],
        'BPR-T': [0.1072, 0.1837, 0.2527, 0.315, 0.3734, 0.4256,
                  0.3543, 0.3486, 0.3453, 0.3409, 0.3365, 0.3312,
                  0.3698, 0.3787, 0.3881, 0.3984, 0.4109, 0.4247,
                  0.5202, 0.5338, 0.5364, 0.5373, 0.5377, 0.5378,
                  6.1069, 6.3375, 6.4003, 6.3218, 6.2361, 6.2296],
        'TGCN': [0.1103, 0.1852, 0.2547, 0.3158, 0.3686, 0.4203,
                 0.3643, 0.3511, 0.3462, 0.3407, 0.3321, 0.3273,
                 0.3853, 0.3874, 0.3953, 0.4044, 0.413, 0.4264,
                 0.5421, 0.5544, 0.5568, 0.5577, 0.5579, 0.558,
                 7.0848, 7.2572, 7.2587, 7.252, 7.2768, 7.2897],
        'LFGCF': [0.1066, 0.188, 0.2637, 0.33, 0.3926, 0.4505,
                  0.3547, 0.3532, 0.3514, 0.3498, 0.3487, 0.3451,
                  0.3708, 0.3825, 0.3948, 0.408, 0.4238, 0.4403,
                  0.5216, 0.5358, 0.5387, 0.5395, 0.5397, 0.54,
                  4.7639, 4.7649, 4.7117, 4.6873, 4.7065, 4.7832],
        'TAGCL': [0.1204, 0.2067, 0.2786, 0.3432, 0.4018, 0.456,
                  0.3917, 0.3841, 0.3777, 0.3705, 0.3647, 0.359,
                  0.4132, 0.4212, 0.4292, 0.4385, 0.4505, 0.4648,
                  0.569, 0.5803, 0.5821, 0.5828, 0.5831, 0.5831,
                  6.2732, 6.039, 5.8107, 5.6063, 5.4543, 5.3548]
    }
}

imporvement = {}
for key in data:
    data_df = pd.DataFrame(data[key])
    data_df[24:] = -data_df[24:]
    data_df['sota'] = data_df.iloc[:, [0, 1, 2, 3, 4]].max(1)
    data_df['imp'] = (data_df['TAGCL'] - data_df['sota']) / data_df['sota']
    data_df[24:] = -data_df[24:]
    temp_list = data_df['imp'].tolist()
    for i in range(len(temp_list)):
        temp_list[i] = str("%.2f%%" % (temp_list[i] * 100))
    imporvement[key] = temp_list

rec_lim = {
    'ml': [0.175, 0.375],
    'lfm': [0.3, 0.6],
    'de': [0.1, 0.5]
}

ndcg_lim = {
    'ml': [0.17, 0.25],
    'lfm': [0.35, 0.55],
    'de': [0.35, 0.5]
}

arp_lim = {
    'ml': [10, 35],
    'lfm': [30, 170],
    'de': [0, 15]
}

dataset_name = {
    'ml': 'Movielens',
    'lfm': 'Last.FM',
    'de': 'Delicious'
}

text_shfit = {
    'ml': {
        'rec': [0.01, 0.007, 0.007, 0.007, 0.007, 0.006],
        'ndcg': [0.003, 0.002, 0.002, 0.002, 0.0017, 0.0017],
        'arp': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5]
    },
    'lfm': {
        'rec': [0.02, 0.015, 0.015, 0.015, 0.015, 0.013],
        'ndcg': [0.007, 0.007, 0.007, 0.007, 0.007, 0.007],
        'arp': [-10, -8, -8, -8, -8, -8]
    },
    'de': {
        'rec': [0.025, 0.02, 0.02, 0.02, 0.02, 0.015],
        'ndcg': [0.007, 0.007, 0.007, 0.007, 0.007, 0.007],
        'arp': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    }
}


def plot_topk():
    x = [5, 10, 15, 20, 25, 30]
    fig = plt.figure(figsize=(15, 13), dpi=400)
    i = 0

    for key in data:
        key_data = data[key]
        ax1 = fig.add_subplot(3, 3, 3*i+1)
        ax1.set_ylim(rec_lim[key])
        ax1.set_xlim(3, 32)
        # ax1.set_title('Recall@K')
        ax1.set_xlabel('K', font_label)
        ax1.set_ylabel('Recall@K', font_label)
        ax1.plot(x, key_data['LGCN'][0:6], label='LGCN', c=colors_map['LGCN'],
                 marker='.', linestyle='-.', linewidth='1.5')
        ax1.plot(x, key_data['SimGCL'][0:6], label='SimGCL', c=colors_map['SimGCL'],
                 marker='.', linestyle='-.', linewidth='1.5')
        ax1.plot(x, key_data['BPR-T'][0:6], label='BPR-T', c=colors_map['BPR-T'],
                 marker='.', linestyle=':', linewidth='1.5')
        ax1.plot(x, key_data['TGCN'][0:6], label='TGCN', c=colors_map['TGCN'],
                 marker='.', linestyle=':', linewidth='1.5')
        ax1.plot(x, key_data['LFGCF'][0:6], label='LFGCF', c=colors_map['LFGCF'],
                 marker='.', linestyle=':', linewidth='1.5')
        ax1.plot(x, key_data['TAGCL'][0:6], label='TAGCL', c=colors_map['TAGCL'],
                 marker='.', linestyle='-', linewidth='2')
        ax1.set_xticks(x)
        x1_label = ax1.get_xticklabels()
        [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        [x1_label_temp.set_fontsize(label_size) for x1_label_temp in x1_label]
        y1_label = ax1.get_yticklabels()
        [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        [y1_label_temp.set_fontsize(label_size) for y1_label_temp in y1_label]
        for a, b, c in zip(x, key_data['TAGCL'][0:6], imporvement[key][0:6]):
            ax1.text(a, b+(text_shfit[key]['rec'][a//5 - 1]), c, ha='center', va='bottom',
                     fontsize=text_size, fontname='Times New Roman')

        ax2 = fig.add_subplot(3, 3, 3*i+2)
        ax2.set_ylim(ndcg_lim[key])
        ax2.set_xlim(3, 32)
        ax2.set_title(dataset_name[key], font_subtitle)
        ax2.set_xlabel('K', font_label)
        ax2.set_ylabel('NDCG@K', font_label)
        ax2.plot(x, key_data['LGCN'][12:18], label='LGCN', c=colors_map['LGCN'],
                 marker='.', linestyle='-.', linewidth='1.5')
        ax2.plot(x, key_data['SimGCL'][12:18], label='SimGCL', c=colors_map['SimGCL'],
                 marker='.', linestyle='-.', linewidth='1.5')
        ax2.plot(x, key_data['BPR-T'][12:18], label='BPR-T', c=colors_map['BPR-T'],
                 marker='.', linestyle=':', linewidth='1.5')
        ax2.plot(x, key_data['TGCN'][12:18], label='TGCN', c=colors_map['TGCN'],
                 marker='.', linestyle=':', linewidth='1.5')
        ax2.plot(x, key_data['LFGCF'][12:18], label='LFGCF', c=colors_map['LFGCF'],
                 marker='.', linestyle=':', linewidth='1.5')
        ax2.plot(x, key_data['TAGCL'][12:18], label='TAGCL', c=colors_map['TAGCL'],
                 marker='.', linestyle='-', linewidth='2')
        ax2.set_xticks(x)
        x2_label = ax2.get_xticklabels()
        [x2_label_temp.set_fontname('Times New Roman') for x2_label_temp in x2_label]
        [x2_label_temp.set_fontsize(label_size) for x2_label_temp in x2_label]
        y2_label = ax2.get_yticklabels()
        [y2_label_temp.set_fontname('Times New Roman') for y2_label_temp in y2_label]
        [y2_label_temp.set_fontsize(label_size) for y2_label_temp in y2_label]
        for a, b, c in zip(x, key_data['TAGCL'][12:18], imporvement[key][12:18]):
            ax2.text(a, b+text_shfit[key]['ndcg'][a//5 - 1], c, ha='center', va='bottom',
                     fontsize=text_size, fontname='Times New Roman')

        ax3 = fig.add_subplot(3, 3, 3*i+3)
        ax3.set_ylim(arp_lim[key])
        ax3.set_xlim(3, 32)
        ax3.set_xlabel('K', font_label)
        ax3.set_ylabel('ARP@K', font_label)
        ax3.plot(x, key_data['LGCN'][24:], label='LGCN', c=colors_map['LGCN'],
                 marker='.', linestyle='-.', linewidth='1.5')
        ax3.plot(x, key_data['SimGCL'][24:], label='SimGCL', c=colors_map['SimGCL'],
                 marker='.', linestyle='-.', linewidth='1.5')
        ax3.plot(x, key_data['BPR-T'][24:], label='BPR-T', c=colors_map['BPR-T'],
                 marker='.', linestyle=':', linewidth='1.5')
        ax3.plot(x, key_data['TGCN'][24:], label='TGCN', c=colors_map['TGCN'],
                 marker='.', linestyle=':', linewidth='1.5')
        ax3.plot(x, key_data['LFGCF'][24:], label='LFGCF', c=colors_map['LFGCF'],
                 marker='.', linestyle=':', linewidth='1.5')
        ax3.plot(x, key_data['TAGCL'][24:], label='TAGCL', c=colors_map['TAGCL'],
                 marker='.', linestyle='-', linewidth='2')
        ax3.set_xticks(x)
        x3_label = ax3.get_xticklabels()
        [x3_label_temp.set_fontname('Times New Roman') for x3_label_temp in x3_label]
        [x3_label_temp.set_fontsize(label_size) for x3_label_temp in x3_label]
        y3_label = ax3.get_yticklabels()
        [y3_label_temp.set_fontname('Times New Roman') for y3_label_temp in y3_label]
        [y3_label_temp.set_fontsize(label_size) for y3_label_temp in y3_label]
        for a, b, c in zip(x, key_data['TAGCL'][24:], imporvement[key][24:]):
            ax3.text(a, b+text_shfit[key]['arp'][a//5 - 1], c, ha='center', va='bottom',
                     fontsize=text_size, fontname='Times New Roman')

        ax3.legend(loc=2, bbox_to_anchor=(0.75, 1.00), borderaxespad=0., fontsize='small',
                   frameon=False, prop={'family': 'Times New Roman', 'size': text_size})
        i += 1

    plt.tight_layout()
    plt.savefig('../outputs/topk_comparison', dpi=400)
    plt.show()
    png1 = io.BytesIO()
    fig.savefig(png1, format="png")
    png2 = Image.open(png1)
    png2.save("../outputs/topk_comparison.tiff")
    png1.close()


if __name__ == '__main__':
    plot_topk()
