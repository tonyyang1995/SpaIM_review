import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from utils import seed_everything
import os
import torch
from options import Options
from dataset import ImputationDataset
from tqdm import tqdm
from SpaIM import ImputeModule
from captum.attr import LayerGradientShap

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
def explain(opt):
    valdataset = ImputationDataset(opt, istrain='val')
    gene_names, cell_names = valdataset.get_eval_names()
    opt.sc_dim = valdataset.get_cluster_dim()

    valdataloader = torch.utils.data.DataLoader(
        valdataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=0
    )

    model = ImputeModule(opt).cuda()

    model.load(os.path.join(opt.save_path, 'last_%d.pth'%(opt.kfold)))

    test_seq = []
    test_st_style = []
    for i, (seq, st_style, spa, seq_cls) in enumerate(valdataloader):
        test_seq.append(seq)
        test_st_style.append(st_style)

    test_seq = torch.cat(test_seq, dim=0)
    test_st_style = torch.cat(test_st_style, dim=0)
    test_inputs = torch.cat([test_seq, test_st_style], dim=1)
    
    # cont 512
    target_layer = model.model.sc_enc1_cont.l
    shap_func = LayerGradientShap(model.model.explain, target_layer)
    baseline_seq = torch.zeros_like(test_seq)#.cuda()
    baseline_style = torch.zeros_like(test_st_style)#.cuda()
    attributions = None
    for i in tqdm(range(0, len(cell_names))):
        attribution = shap_func.attribute(
            (test_seq, test_st_style),
            baselines=(baseline_seq, baseline_style),
            target=i
        )
        if attributions is None:
            attributions = attribution
        else:
            attributions = attributions + attribution
    
    # normalize the shape values
    attributions = attributions / len(cell_names)
    print("attributions: ", attributions.shape, len(gene_names), len(cell_names))
    df = pd.DataFrame(attributions.T.cpu().numpy(), columns=gene_names)
    df.to_pickle(os.path.join(opt.save_path, "cont_gradshap_512_%d.pkl"%(opt.kfold)))

    # cont 256
    target_layer = model.model.sc_enc2_cont.l
    shap_func = LayerGradientShap(model.model.explain, target_layer)
    baseline_seq = torch.zeros_like(test_seq)#.cuda()
    baseline_style = torch.zeros_like(test_st_style)#.cuda()
    attributions = None
    for i in tqdm(range(0, len(cell_names))):
        attribution = shap_func.attribute(
            (test_seq, test_st_style),
            baselines=(baseline_seq, baseline_style),
            target=i
        )
        if attributions is None:
            attributions = attribution
        else:
            attributions = attributions + attribution
    
    # normalize the shape values
    attributions = attributions / len(cell_names)
    print("attributions: ", attributions.shape, len(gene_names), len(cell_names))
    df = pd.DataFrame(attributions.T.cpu().numpy(), columns=gene_names)
    df.to_pickle(os.path.join(opt.save_path, "cont_gradshap_256_%d.pkl"%(opt.kfold)))

    # style 512
    target_layer = model.model.enc_style1.l
    shap_func = LayerGradientShap(model.model.explain, target_layer)
    baseline_seq = torch.zeros_like(test_seq).cuda()
    baseline_style = torch.zeros_like(test_st_style).cuda()
    attributions = None
    for i in tqdm(range(0, len(cell_names))):
        attribution = shap_func.attribute(
            (test_seq.cuda(), test_st_style.cuda()),
            baselines=(baseline_seq, baseline_style),
            target=i
        )
        if attributions is None:
            attributions = attribution
        else:
            attributions = attributions + attribution
    
    # normalize the shape values
    attributions = attributions / len(cell_names)
    print("attributions: ", attributions.shape, len(gene_names), len(cell_names))
    df = pd.DataFrame(attributions.T.cpu().numpy(), columns=gene_names)
    df.to_pickle(os.path.join(opt.save_path, "style_gradshap_512_%d.pkl"%(opt.kfold)))

    # style 256
    target_layer = model.model.enc_style2.l
    shap_func = LayerGradientShap(model.model.explain, target_layer)
    baseline_seq = torch.zeros_like(test_seq).cuda()
    baseline_style = torch.zeros_like(test_st_style).cuda()
    attributions = None
    for i in tqdm(range(0, len(cell_names))):
        attribution = shap_func.attribute(
            (test_seq.cuda(), test_st_style.cuda()),
            baselines=(baseline_seq, baseline_style),
            target=i
        )
        if attributions is None:
            attributions = attribution
        else:
            attributions = attributions + attribution
    
    # normalize the shape values
    attributions = attributions / len(cell_names)
    print("attributions: ", attributions.shape, len(gene_names), len(cell_names))
    df = pd.DataFrame(attributions.T.cpu().numpy(), columns=gene_names)
    df.to_pickle(os.path.join(opt.save_path, "style_gradshap_256_%d.pkl"%(opt.kfold)))

def val(opt):
    valdataset = ImputationDataset(opt, istrain='val')
    gene_names, cell_names = valdataset.get_eval_names()
    opt.sc_dim = valdataset.get_cluster_dim()

    valdataloader = torch.utils.data.DataLoader(
        valdataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=0
    )

    model = ImputeModule(opt)
    if opt.parallel:
        model = torch.nn.DataParallel(model).cuda().module
    else:
        model = model.cuda()
    model.load(os.path.join(opt.save_path, 'last_%d.pth'%(opt.kfold)))

    with torch.no_grad():
        eval_result = None
        input_result = None
        for i, (seq, st_style, spa, seq_cls) in enumerate(valdataloader):
            inputs = {
                'scx': seq,
                'st_style': st_style,
                'sc_cls': seq_cls
            }
        model.set_input(inputs, istrain=0)
        out = model.inference()
        impute_result = out['st_fake'].detach().cpu().numpy()

        eval_result = impute_result if eval_result is None else np.concatenate((eval_result, impute_result), axis=0)
        input_result = spa.detach().cpu().numpy() if input_result is None else np.concatenate((input_result, spa.detach().cpu().numpy()), axis=0)

    eval_result = eval_result.T
    eval_result[eval_result <0] = 0
    input_result = input_result.T
    
    df1 = pd.DataFrame(eval_result, index=cell_names, columns=gene_names)
    df1.to_pickle(os.path.join(opt.save_path, 'impute_result_%d.pkl'%(opt.kfold)))

    df2 = pd.DataFrame(input_result, index=cell_names, columns=gene_names)
    df2.to_pickle(os.path.join(opt.save_path, 'input_result_%d.pkl'%(opt.kfold)))

    # np.save(os.path.join(opt.save_path, 'impute_result_%d.npy'%(opt.kfold)), eval_result)
    # np.save(os.path.join(opt.save_path, 'input_result_%d.npy'%(opt.kfold)), input_result)
    # evaluate = CalculateMeteics(df2, df1, opt.save_path, 'None', 'SpaImputation')
    # acc = evaluate.compute_all(i)
    # print(acc.T['PCC'].mean(), acc.T['JS'].mean())

def Data_augmentation(data1, data2, data3, times=2, zero_fraction=0.1):
    """
    参数:
    - data1: 第一个数据张量，torch.Tensor
    - times: 增强的倍数
    - zero_fraction: 被置为0的数据比例，默认为0.1
    
    返回:
    - augmented_data1: 增强后的第一个数据张量
    """
    
    # 初始化列表，用于存储原始数据和所有增强后的数据
    all_data = [data1]
    
    # 对数据进行三次增强
    for _ in range(times-1):
        # 生成与数据张量形状相同的随机掩码
        device = data1.device  # 确保掩码在同一设备上
        mask = torch.rand(data1.shape, device=device) < zero_fraction  # True 的概率为 zero_fraction
        
        # 应用掩码，将选定的元素置为0
        augmented_data = torch.where(mask, torch.zeros_like(data1), data1)
        
        # 将增强后的数据添加到列表中
        all_data.append(augmented_data)
    
    # 将原始数据和所有增强数据叠加
    concatenated_data1 = torch.cat(all_data, dim=0)
    concatenated_data2 = torch.cat([data2] * times, dim=0)
    concatenated_data3 = torch.cat([data3] * times, dim=0)
    
    return concatenated_data1, concatenated_data2, concatenated_data3

def train(opt):
    dataset = ImputationDataset(opt, istrain='train')
    valdataset = ImputationDataset(opt, istrain='val')

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=0
    )
    valdataloader = torch.utils.data.DataLoader(
        valdataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=0
    )

    opt.sc_dim = dataset.get_cluster_dim()

    model = ImputeModule(opt)
    if opt.parallel:
        model = torch.nn.DataParallel(model).cuda().module
    else:
        model = model.cuda()    

    tqdm_bar = tqdm(range(opt.epochs))
    for epoch in tqdm_bar:
        for i, (seq, spa, sc_style, st_style, seq_cls) in enumerate(dataloader):
            # data enhance
            seq, spa, st_style = Data_augmentation(seq, spa, st_style, times=4, zero_fraction=0.5)  # Selective enhancement of some datasets
            inputs = {
                'scx': seq,
                'stx': spa,
                'sc_style': sc_style,
                'st_style': st_style,
                'sc_cls': seq_cls  # 其实没有用
            }

            model.set_input(inputs, istrain=1)
            model.update_parameters()        # 走完整个train的流程
            loss_stat = model.get_current_loss()
            msg = 'Epoch: {}/{}, Iter: {}/{}, '.format(epoch, opt.epochs, i, len(dataloader))
            for k,v in loss_stat.items():
                msg += ' %s: %.4f' % (k, v)
    
    model.save(os.path.join(opt.save_path, 'last_%d.pth'%(opt.kfold)))


if __name__ == '__main__':
    opt = Options().parse()
    seed_everything(opt.seed)
    if opt.val_only == 0:
        train(opt)
        # val(opt)
        # explain(opt)
    else:
        val(opt)
    explain(opt)

    torch.cuda.empty_cache()
