在这里记录我们可能需要比较的baseline和benchmark

## Victim Model

bert-base-uncased

## Attack

token/word-level attacks: BadNets, AddSent, LWP, SOS

syntactic/semantic based backdoors: HiddenKiller, StyleBkd

## Baseline Defense

ONION，STRIP，RAP

## DataSet

SST-2：情感分析

Offenseval：有毒物质检测

AG's News：主题分类

## BenchMark

ΔASR(Attack Success Rate)：攻击成功率降低大小，越大越好。

ΔCA(Clean accuracy)：干净准确率降低大小，越小越好。

FRR(False Rcceptance Rate)：错误拒绝率，中毒样本被错误标记为干净样本的百分比，越小越好。

FAR(False Acceptance Rate)：错误接受率，干净样本被错误标记为中毒样本的百分比，越小越好。
