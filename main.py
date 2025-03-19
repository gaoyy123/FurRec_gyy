import math
import random
import torch
import numpy as np
from time import time
from prettytable import PrettyTable
import torch.optim.lr_scheduler as lr_scheduler
from utility.parser_Metakg import parse_args
from utility.data_loader import load_data_cold
from model.Fuzzy import Recommender
from utility.evaluate import test
from utility.helper import early_stopping
from utility.scheduler import Scheduler
from tqdm import tqdm

torch.cuda.empty_cache()

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0
sample_num = 10


def get_feed_dict_cold(support_user_set, support_user_neg):
    support_meta_set = []
    for (key, val), (_, neg) in zip(support_user_set.items(), support_user_neg.items()):
        feed_dict = []
        sample_num = len(val)
        user = [int(key)] * sample_num
        if len(val) != sample_num:
            pos_item = np.random.choice(list(val), sample_num, replace=True)
        else:
            pos_item = val
        if len(neg) < sample_num:
            while len(neg) == sample_num:
                i = np.random.randint(low=0, high=n_items, size=1)[0]
                if i not in (neg and val):
                    continue
                neg.append(i)
        else:
            neg_item = neg[:sample_num]
        feed_dict.append(np.array(user))
        feed_dict.append(np.array(list(pos_item)))
        feed_dict.append(np.array(neg_item))
        support_meta_set.append(feed_dict)

    return support_meta_set  # [n_user, 3, 10]

def get_feed_kg(kg_graph):
    triplet_num = len(kg_graph)
    pos_hrt_id = np.random.randint(low=0, high=triplet_num, size=args.batch_size * sample_num)
    pos_hrt = kg_graph[pos_hrt_id]
    neg_t = np.random.randint(low=0, high=n_entities, size=args.batch_size * sample_num)

    return torch.LongTensor(pos_hrt[:, 0]).to(device), torch.LongTensor(pos_hrt[:, 1]).to(device), torch.LongTensor(
        pos_hrt[:, 2]).to(device), torch.LongTensor(neg_t).to(device)

def convert_to_sparse_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape).to(device)

def get_net_parameter_dict(params):
    param_dict = dict()
    indexes = []
    for i, (name, param) in enumerate(params):
        if param.requires_grad:
            param_dict[name] = param.to(device)
            indexes.append(i)

    return param_dict, indexes

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    Ks = eval(args.Ks)

    """build dataset"""
    _, _, user_dict, user_neg, n_params, graph, mat_list = load_data_cold(args, 'meta_training')
    _, mean_mat_list = mat_list


    print('Complete data loading')
    src, dst = graph.edges()
    type = graph.edata['type']
    kg_graph = np.array(list(zip(src.numpy(), type.numpy(), dst.numpy())))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """use pretrain data"""
    if args.use_pretrain:
        pre_path = args.data_path + 'pretrain/{}/mf.npz'.format(args.dataset)
        pre_data = np.load(pre_path)
        user_pre_embed = torch.tensor(pre_data['user_embed'])
        item_pre_embed = torch.tensor(pre_data['item_embed'])
    else:
        user_pre_embed = None
        item_pre_embed = None

    """init model"""
    print('model:CKG')
    model = Recommender(n_params, args, graph, user_pre_embed, item_pre_embed)
    model = torch.nn.DataParallel(model)
    model.module.to(device)
    names_weights_copy, indexes = get_net_parameter_dict(model.named_parameters())
    scheduler = Scheduler(len(names_weights_copy), device, grad_indexes=indexes).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_update_lr)
    optimize = lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.meta_gamma)
    scheduler_optimizer = torch.optim.Adam(scheduler.parameters(), lr=args.scheduler_lr)
    scheduler_optimize = lr_scheduler.StepLR(scheduler_optimizer, step_size=10, gamma=0.98)

    """prepare feed data"""
    support_meta_set = get_feed_dict_cold(user_dict['train_user_set'],user_neg['train_user_set_neg'])
    query_meta_set = get_feed_dict_cold(user_dict['test_user_set'],user_neg['test_user_set_neg'])

    # shuffle
    index = np.arange(len(support_meta_set))
    np.random.shuffle(index)
    support_meta_set = [support_meta_set[i] for i in index]
    query_meta_set = [query_meta_set[i] for i in index]

    interact_mat = convert_to_sparse_tensor(mean_mat_list)
    del mean_mat_list

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    print('start training')
    if args.use_meta_model:
        model.load_state_dict(torch.load('./model_para/meta_model_{}.ckpt'.format(args.dataset)))
    else:
        model.module.train()
        model.module.interact_mat = interact_mat

        sum_loss = 0
        iter_num = min(math.ceil(len(user_dict['train_user_set']) / args.batch_size), 300)
        train_s_t = time()
        for s in tqdm(range(iter_num)):
            # for s in tqdm(range(1)):
            try:
                batch_support = support_meta_set[s * args.batch_size:(s + 1) * args.batch_size]
                batch_query = query_meta_set[s * args.batch_size:(s + 1) * args.batch_size]
            except IndexError:
                continue
            pt = int(s / iter_num * 100)
            if len(batch_support) > args.meta_batch_size:
                task_losses, weight_meta_batch = scheduler.get_weight(batch_support, batch_query, model, pt)
                torch.cuda.empty_cache()
                task_prob = torch.softmax(weight_meta_batch.reshape(-1), dim=-1)
                selected_tasks_idx = scheduler.sample_task(task_prob, args.meta_batch_size)
                selected_tasks_idx = selected_tasks_idx.tolist()
                batch_support = [batch_support[i] for i in selected_tasks_idx]
                batch_query = [batch_query[i] for i in selected_tasks_idx]

            selected_losses = scheduler.compute_loss(batch_support, batch_query, model)
            meta_batch_loss = torch.mean(selected_losses)

            h, r, pos_t, neg_t = get_feed_kg(kg_graph)
            kg_loss = model.module.forward_kg(h, r, pos_t, neg_t)
            del h, r, pos_t, neg_t
            torch.cuda.empty_cache()
            batch_loss = kg_loss + meta_batch_loss

            """update network"""
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            optimize.step()

            """update scheduler"""
            loss_scheduler = 0
            for (su, qu) in zip(batch_support, batch_query):
                loss_scheduler += model.module.forward_meta(su, qu, model.module.get_parameter())

            scheduler_optimizer.zero_grad()
            loss_scheduler.backward()
            scheduler_optimizer.step()
            scheduler_optimize.step()

            torch.cuda.empty_cache()
        if args.save:
            torch.save(model.state_dict(), args.out_dir + 'meta_model_' + args.dataset +'.ckpt')

        train_e_t = time()
        print('meta_training_time: ', train_e_t - train_s_t)

    '''fine tune'''
    _, _, cold_user_dict, cold_user_neg, cold_n_params, _, cold_mat_list = load_data_cold(args)
    _, cold_mean_mat_list = cold_mat_list
    cold_interact_mat = convert_to_sparse_tensor(cold_mean_mat_list)
    model.module.interact_mat = cold_interact_mat
    # reset lr
    for g in optimizer.param_groups:
        g['lr'] = args.lr

    train_supp = cold_user_dict['train_user_set']
    train_supp_neg = cold_user_neg['train_user_set_neg']
    test_qry = cold_user_dict['test_user_set']
    test_qry_neg = cold_user_neg['test_user_set_neg']
    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    print("start fine tune...")
    epoch = 0

    data = get_feed_dict_cold(train_supp, train_supp_neg)
    data_q = get_feed_dict_cold(test_qry, test_qry_neg)
    u_batch_size = 32
    test_users = list(train_supp.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    while (1):
        # bprloss + recallloss
        epoch += 1
        model.module.train()
        loss = 0
        train_s_t = time()

        indices = np.random.permutation(len(data))
        data = [data[i] for i in indices]
        data_q = [data_q[i] for i in indices]
        for u_batch_id in tqdm(range(n_user_batchs)):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            batch = data[start:end]
            batch_q = data_q[start:end]
            pos_q = [sublist[1] for sublist in batch_q]
            batch_loss = model.module.forward_cold(batch, pos_q)
            batch_loss = batch_loss / u_batch_size
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
        print(loss)
        train_e_t = time()

        if epoch % args.test_step == 0 or epoch == 1:
            model.module.eval()
            torch.cuda.empty_cache()
            test_s_t = time()
            with torch.no_grad():
                ret = test(model, cold_user_dict, cold_n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, ret['recall'], ret['ndcg']])
            print(train_res)
            f = open('./result/{}_{}_bt{}_lr{}_metaLr{}.txt'.format(args.dataset,
                                                                    args.fine_tune_batch_size, args.lr,
                                                                    args.meta_update_lr), 'a+')
            f.write(str(train_res) + '\n')
            f.close()
            # early stopping.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=20)
            if should_stop:
                break
        else:
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss))
        if epoch == args.epoch:
            quit()
    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))