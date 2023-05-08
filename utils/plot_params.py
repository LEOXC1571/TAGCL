# -*- coding: utf-8 -*-
# @Filename: plots
# @Date: 2022-06-09 18:32
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import io
import numpy as np
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

de_tagcl_nl_rec = [0.3403, 0.3329, 0.3432, 0.3303, 0.3383]
de_tagcl_nl_arp = [4.9465, 5.8791, 5.6063, 5.2608, 5.5091]

de_tagcl_es_rec = [0.3219, 0.3432, 0.3299, 0.3265, 0.3436]
de_tagcl_es_arp = [6.9805, 5.6063, 4.9995, 5.1348, 4.8663]

lfm_tagcl_nl_rec = [0.5209, 0.5199, 0.5197, 0.1629, 0.1800]
lfm_tagcl_nl_arp = [47.1529, 42.9929, 44.0176, 31.9133, 40.0645]

lfm_tagcl_es_rec = [0.4739, 0.5199, 0.5287, 0.5306, 0.5330]
lfm_tagcl_es_arp = [45.9634, 42.9929, 46.1464, 41.8205, 42.0945]

text_shift = [0.007, 0.007, 0.007, 0.007, 0.007,
              -0.025, -0.03, -0.025, -0.025, -0.025,
              0.001, -0.004, 0.001, -0.004, 0.001,
              -0.004, 0.001, -0.004, -0.004, 0.001]


def plot_params():
    x = np.arange(1, 6)

    fig = plt.figure(figsize=(16, 4), dpi=400)

    ax1 = fig.add_subplot(141)
    ax1.bar(x, lfm_tagcl_nl_arp, label='ARP@20', color=colors[1], alpha=0.7, width=0.5)
    ax1.set_ylim([30, 50])
    ax1.set_title('TAGCL on Last.FM with various numbers of layers', font_subtitle)
    ax1.set_xlabel('Number of layers', font_label)
    ax1.set_ylabel('ARP@20', font_label)
    ax11 = ax1.twinx()
    ax11.set_ylabel('Recall@20', font_label)
    ax11.plot(x, lfm_tagcl_nl_rec, label='Recall@20', c='cornflowerblue', marker='.', linewidth='2')
    ax11.set_ylim([0.15, 0.55])
    x1_label = ax1.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    [x1_label_temp.set_fontsize(label_size) for x1_label_temp in x1_label]
    y1_label = ax1.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    [y1_label_temp.set_fontsize(label_size) for y1_label_temp in y1_label]
    y11_label = ax11.get_yticklabels()
    [y11_label_temp.set_fontname('Times New Roman') for y11_label_temp in y11_label]
    [y11_label_temp.set_fontsize(label_size) for y11_label_temp in y11_label]
    for a, b in zip(x, lfm_tagcl_nl_rec):
        ax11.text(a, b + text_shift[a-1], b, ha='center', va='bottom', fontsize=text_size, fontname='Times New Roman')
    ax1.legend(loc=2, bbox_to_anchor=(0.58, 1.00), borderaxespad=0., fontsize='small',
               frameon=False, prop={'family': 'Times New Roman', 'size': text_size})
    ax11.legend(loc=2, bbox_to_anchor=(0.58, 0.93), borderaxespad=0., fontsize='small',
               frameon=False, prop={'family': 'Times New Roman', 'size': text_size})

    ax2 = fig.add_subplot(142)
    ax2.bar(x, lfm_tagcl_es_arp, label='ARP@20', color=colors[2], alpha=0.7, width=0.5)
    ax2.set_ylim([30, 50])
    ax2.set_title('TAGCL on Last.FM with various embedding sizes', font_subtitle)
    ax2.set_xlabel('Emebdding size', font_label)
    ax2.set_ylabel('ARP@20', font_label)
    ax21 = ax2.twinx()
    ax21.set_ylabel('Recall@20', font_label)
    ax21.plot(x, lfm_tagcl_es_rec, label='Recall@20', c='mediumpurple', marker='.', linewidth='2')
    ax21.set_ylim([0.15, 0.55])
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xticklabels([32, 64, 128, 256, 512])
    x2_label = ax2.get_xticklabels()
    [x2_label_temp.set_fontname('Times New Roman') for x2_label_temp in x2_label]
    [x2_label_temp.set_fontsize(label_size) for x2_label_temp in x2_label]
    y2_label = ax2.get_yticklabels()
    [y2_label_temp.set_fontname('Times New Roman') for y2_label_temp in y2_label]
    [y2_label_temp.set_fontsize(label_size) for y2_label_temp in y2_label]
    y21_label = ax21.get_yticklabels()
    [y21_label_temp.set_fontname('Times New Roman') for y21_label_temp in y21_label]
    [y21_label_temp.set_fontsize(label_size) for y21_label_temp in y21_label]
    for a, b in zip(x, lfm_tagcl_es_rec):
        ax21.text(a, b + text_shift[a+4], b, ha='center', va='bottom', fontsize=text_size, fontname='Times New Roman')
    ax2.legend(loc=2, bbox_to_anchor=(0.58, 0.90), borderaxespad=0., fontsize='small',
               frameon=False, prop={'family': 'Times New Roman', 'size': text_size})
    ax21.legend(loc=2, bbox_to_anchor=(0.58, 0.83), borderaxespad=0., fontsize='small',
               frameon=False, prop={'family': 'Times New Roman', 'size': text_size})

    ax3 = fig.add_subplot(143)
    ax3.bar(x, de_tagcl_nl_arp, label='ARP@20', color=colors[1], alpha=0.7, width=0.5)
    ax3.set_ylim([4.5, 7.5])
    ax3.set_title('TAGCL on Delicious with various numbers of layers', font_subtitle)
    ax3.set_xlabel('Number of layers', font_label)
    ax3.set_ylabel('ARP@20', font_label)
    ax31 = ax3.twinx()
    ax31.set_ylabel('Recall@20', font_label)
    ax31.plot(x, de_tagcl_nl_rec, label='Recall@20', c='cornflowerblue', marker='.', linewidth='2')
    ax31.set_ylim([0.3, 0.35])
    x3_label = ax3.get_xticklabels()
    [x3_label_temp.set_fontname('Times New Roman') for x3_label_temp in x3_label]
    [x3_label_temp.set_fontsize(label_size) for x3_label_temp in x3_label]
    y3_label = ax3.get_yticklabels()
    [y3_label_temp.set_fontname('Times New Roman') for y3_label_temp in y3_label]
    [y3_label_temp.set_fontsize(label_size) for y3_label_temp in y3_label]
    y31_label = ax31.get_yticklabels()
    [y31_label_temp.set_fontname('Times New Roman') for y31_label_temp in y31_label]
    [y31_label_temp.set_fontsize(label_size) for y31_label_temp in y31_label]
    for a, b in zip(x, de_tagcl_nl_rec):
        ax31.text(a, b + text_shift[a+9], b, ha='center', va='bottom', fontsize=text_size, fontname='Times New Roman')
    ax3.legend(loc=2, bbox_to_anchor=(0.58, 1.00), borderaxespad=0., fontsize='small',
               frameon=False, prop={'family': 'Times New Roman', 'size': text_size})
    ax31.legend(loc=2, bbox_to_anchor=(0.58, 0.93), borderaxespad=0., fontsize='small',
               frameon=False, prop={'family': 'Times New Roman', 'size': text_size})

    ax4 = fig.add_subplot(144)
    ax4.bar(x, de_tagcl_es_arp, label='ARP@20', color=colors[2], alpha=0.7, width=0.5)
    ax4.set_ylim([4.5, 7.5])
    ax4.set_title('TAGCL on Delicious with various embedding sizes', font_subtitle)
    ax4.set_xlabel('Embedding size', font_label)
    ax4.set_ylabel('ARP@20', font_label)
    ax41 = ax4.twinx()
    ax41.set_ylabel('Recall@20', font_label)
    ax41.plot(x, de_tagcl_es_rec, label='Recall@20', c='mediumpurple', marker='.', linewidth='2')
    ax41.set_ylim([0.3, 0.35])
    ax4.set_xticks([1, 2, 3, 4, 5])
    ax4.set_xticklabels([32, 64, 128, 256, 512])
    x4_label = ax4.get_xticklabels()
    [x4_label_temp.set_fontname('Times New Roman') for x4_label_temp in x4_label]
    [x4_label_temp.set_fontsize(label_size) for x4_label_temp in x4_label]
    y4_label = ax4.get_yticklabels()
    [y4_label_temp.set_fontname('Times New Roman') for y4_label_temp in y4_label]
    [y4_label_temp.set_fontsize(label_size) for y4_label_temp in y4_label]
    y41_label = ax41.get_yticklabels()
    [y41_label_temp.set_fontname('Times New Roman') for y41_label_temp in y41_label]
    [y41_label_temp.set_fontsize(label_size) for y41_label_temp in y41_label]
    for a, b in zip(x, de_tagcl_es_rec):
        ax41.text(a, b + text_shift[a+14], b, ha='center', va='bottom', fontsize=text_size, fontname='Times New Roman')
    ax4.legend(loc=2, bbox_to_anchor=(0.38, 1.00), borderaxespad=0., fontsize='small',
               frameon=False, prop={'family': 'Times New Roman', 'size': text_size})
    ax41.legend(loc=2, bbox_to_anchor=(0.38, 0.93), borderaxespad=0., fontsize='small',
               frameon=False, prop={'family': 'Times New Roman', 'size': text_size})

    plt.tight_layout()
    plt.savefig('../outputs/tagcl_param', dpi=400)
    plt.show()
    png1 = io.BytesIO()
    fig.savefig(png1, format="png")
    png2 = Image.open(png1)
    png2.save("../outputs/tagcl_param.tiff")
    png1.close()


if __name__ == '__main__':
    plot_params()

