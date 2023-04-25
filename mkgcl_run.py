import argparse
from recbole.quick_start import run_recbole
import uuid


def main(model_name, dataset_name, parameter_dict, config_file=None):
    # 1.set param
    parser = argparse.ArgumentParser()
    # set model # MKGCL,KGAT,SGL,LightGCN,RippleNet
    parser.add_argument('--model', '-m', type=str, default=model_name, help='name of models')
    # set datasets # ml-1m,ml-20m,amazon-books,lfm1b-tracks
    parser.add_argument('--dataset', '-d', type=str, default=dataset_name, help='name of datasets')
    # set config
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    args, _ = parser.parse_known_args()
    yaml_dict = {'baseline': {'ml-1m': 'zone/baseline_ml.yaml', 'ml-20m': 'zone/baseline_ml.yaml'
        , 'amazon-books': 'zone/baseline_amazon.yaml', 'lfm1b-tracks': 'zone/baseline_lfm1b.yaml'},
                 'mkgcl': {'ml-1m': 'zone/mkgcl_ml.yaml', 'ml-20m': 'zone/mkgcl_ml.yaml'
                     , 'amazon-books': 'zone/mkgcl_amazon.yaml', 'lfm1b-tracks': 'zone/mkgcl_lfm1b.yaml'}
                 }

    if config_file == None:
        if model_name == 'MKGCL':
            config_file = yaml_dict['mkgcl'][dataset_name]
        else:
            config_file = yaml_dict['baseline'][dataset_name]

    config_file_list = [config_file]

    print("running_flag: ", parameter_dict['running_flag'])
    print('config_file: ', config_file)

    # 2.call recbole: config,dataset,model,trainer,training,evaluation
    result = run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list,
                         config_dict=parameter_dict)

    # 3.print result
    # print(result)


if __name__ == '__main__':
    # time.sleep(3600 * 2)

    # param dict
    parameter_dict = {
        # 'neg_sampling': None,
        'running_flag': str(uuid.uuid4()),
        'epochs': 500,
        'train_batch_size': 4096,  # 4096
        'open_migcl': True,  # alignment
        'open_mulcl': False,  # uniformity
        'open_ali_uni': False,
        'open_represent': False,
        'open_r': False,
        'migcl_data_aug': 'sen_a',  # ed;gen_a:ed,gen_m:ed,gen_agen_m;sen_a,sen_m
        'mulcl_data_aug': 'gen_a',  # sen_a,sen_m,gen_a,gen_m
        'r_data_aug': 'r_s',  # sen_a,sen_m,r_s(Relation Substitute)
        'mini_batch_size_mul': 32,  # 32
        'noise_base_a': 10,
        'noise_base_m': 1e-3,
        'u_noise_base': 1e-3,
        'e_noise_base': 1e-3,
        'r_noise_base': 1e-4,
        'r_s_len': 1e-2,
        'temperature1': 1.0,  # 1.0,0.75,0.5,0.25
        'temperature2': 1.0,  # 1.0,0.75,0.5,0.25
        # 'data_path': 'dataset/',
        'align_w': 1,  # Alignment loss weight
        'unif_w': 1,  # Uniformity loss weight
        'align_alpha': 2,  # alpha in alignment loss
        'unif_t': 2,  # t in uniformity loss
        'log_interval': 10,  # Number of iterations between logs
    }
    # model: MKGCL,KGAT,BPR,NGCF,KGIN,KGCL,FM,NFM,CKE,CFKG,KGCN
    # datasets: ml-1m,amazon-books,lfm1b-tracks

    model_name = 'MKGCL'
    dataset_name = 'amazon-books'
    main(model_name, dataset_name, parameter_dict)
    # dataset_name = 'lfm1b-tracks'
    # main(model_name, dataset_name, parameter_dict)
    # dataset_name = 'ml-1m'
    # main(model_name, dataset_name, parameter_dict)

