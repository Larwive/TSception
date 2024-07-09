import torch.backends.mps

from utils import *
import copy
import torch.nn as nn
from captum.attr import IntegratedGradients
import torch

torch.set_printoptions(profile="full")

CUDA = torch.cuda.is_available()
MPS = torch.backends.mps.is_available()

def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if MPS:
            x_batch, y_batch = x_batch.to('mps'), y_batch.to('mps')
        elif CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        out = net(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train


def predict(data_loader, net, loss_fn, contributions=None, max_contributions=None, enable_contribution=False, total=0):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    if enable_contribution:
        ig = IntegratedGradients(net)
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            #ig = IntegratedGradients(net.to('cpu'))
            #attributions, delta = ig.attribute(x_batch.detach().clone().to('cpu'),
                                               #target=y_batch.detach().clone().to('cpu'), return_convergence_delta=True)

            if MPS:
                x_batch, y_batch, net = x_batch.to('mps'), y_batch.to('mps'), net.to('mps')
            elif CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            if enable_contribution:
                attributions, delta = ig.attribute(x_batch,
                                               target=y_batch, return_convergence_delta=True)
                total = get_contribution(attributions, contributions, max_contributions, total)
                print(total)
                print("Contributions\n", [contribution / total for contribution in contributions])
                print("Max contribution\n", max_contributions)

            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val, total


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, data_val, label_val, subject, fold):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_trial' + str(fold)
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)

    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    model = get_model(args)
    para = get_trainable_parameter_num(model)
    print('Model {} size:{}'.format(args.model, para))

    if MPS:
        model = model.to('mps')
    elif CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):

        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val, total = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))


        if acc_val > trlog['max_acc']:
            trlog['max_acc'] = acc_val
            save_model('max-acc{}'.format(args.label_type))

            if args.save_model:
                # save model here for reproduce
                model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
                data_type = 'model_{}_{}_{}'.format(args.dataset, args.data_format, args.label_type)
                save_path = osp.join(args.save_path, data_type)
                ensure_path(save_path)
                model_name_reproduce = osp.join(save_path, model_name_reproduce)
                torch.save(model.state_dict(), model_name_reproduce)

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))
    save_name_ = 'trlog' + save_name
    ensure_path(osp.join(args.save_path, 'log_train'))
    torch.save(trlog, osp.join(args.save_path, 'log_train', save_name_))

    return trlog['max_acc']


def test(args, data, label, reproduce, subject, fold, contributions=None, max_contributions=None, total=0, model=None):
    seed_all(args.random_seed)
    set_up(args)

    test_loader = get_dataloader(data, label, args.batch_size, False)

    if model is None:
        model = get_model(args)
        if MPS:
            model = model.to('mps')
        elif CUDA:
            model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()


    if reproduce:
        model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
        data_type = 'model_{}_{}_{}'.format(args.dataset, args.data_format, args.label_type)
        save_path = osp.join(args.save_path, data_type)
        ensure_path(save_path)
        model_name_reproduce = osp.join(save_path, model_name_reproduce)
        model.load_state_dict(torch.load(model_name_reproduce, map_location=torch.device('mps')))
    else:
        model.load_state_dict(torch.load(args.load_path.format(args.label_type)))
    loss, pred, act, total = predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn, contributions=contributions, max_contributions=max_contributions, enable_contribution=args.contribution, total=total
    )
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act, total


