import logging
import os

import torch
from tqdm import tqdm
import utils
from evaluate import evaluate
from collections import OrderedDict
def get_lr(opt):

    return opt.param_groups[0]['lr']

def inference(model, batch, device=None):
    model.eval()

    with torch.no_grad():
        y_pred = model(batch.to(device))
        y_pred = y_pred["out"]

    return y_pred

def train_epoch(model,
                loss_fn,
                dataset_dl,
                opt=None,
                lr_scheduler=None,
                metrics=None,
                params=None,
                device=None):

    running_loss = utils.RunningAverage()

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metric.reset()

    for (xb, yb) in tqdm(dataset_dl):
        xb = xb.to(device)
        yb = yb.to(device)

        output = model(xb)['out']
        loss_b = loss_fn(output, yb)

        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss.update(loss_b.item())

        if metrics is not None:
            for metric_name, metric in metrics.items():
                metric.add(output.detach(), yb)

    if metrics is not None:
        metrics_results = OrderedDict({})

        for metric_name, metric in metrics.items():
            metrics_results[metric_name] = metric.value()

        return running_loss(), metrics_results

    else:

        return running_loss(), None


def train_and_evaluate(model,
                       train_dataloader,
                       val_dataloader,
                       opt,
                       loss_fn,
                       metrics,
                       params,
                       lr_scheduler,
                       ckpt_dir,
                       ckpt_filename,
                       log_dir,
                       writer,
                       device=None):

    ckpt_file_path = os.path.join(ckpt_dir, ckpt_filename)
    best_value = -float('inf')
    early_stopping = utils.EarlyStopping(patience=10, verbose=True)
    start_epoch = 0

    batch_sample_train, batch_gt_train = next(iter(train_dataloader))
    batch_sample_val, batch_gt_val = next(iter(val_dataloader))

    if os.path.exists(ckpt_file_path):
        model, opt, lr_scheduler, start_epoch, best_value = \
            utils.load_checkpoint(model,
                                  opt,
                                  lr_scheduler,
                                  start_epoch,
                                  False,
                                  best_value,
                                  ckpt_dir,
                                  ckpt_filename)
        print("=> loaded checkpoint form {} (epoch {})".format(
            ckpt_file_path, start_epoch))
    else:
        print("=> Initializing from scratch")

    for epoch in range(start_epoch, params.n_epochs):
        # Run one epoch
        current_lr = get_lr(opt)
        logging.info('Epoch {}/{}, current lr={}'.format(epoch,
                                                         params.n_epochs - 1,
                                                         current_lr))
        writer.add_scalar('Learning_rate', current_lr, epoch)

        model.train()
        train_loss, train_metrics = train_epoch(model,
                                                loss_fn,
                                                train_dataloader,
                                                opt,
                                                lr_scheduler,
                                                metrics,
                                                params,
                                                device=device)

        # Evaluate for one epoch on validation set
        val_loss, val_metrics = evaluate(model,
                                         loss_fn,
                                         val_dataloader,
                                         metrics=metrics,
                                         device=device)

        writer.add_scalars('Loss', {
            'Training': train_loss,
            'Validation': val_loss,
        }, epoch)

        for (train_metric_name, train_metric_results), \
            (val_metric_name, val_metric_results) in zip(train_metrics.items(),
                                                         val_metrics.items()):
            writer.add_scalars(train_metric_name, {
                'Training': train_metric_results[0],
                'Validation': val_metric_results[0],
            }, epoch)

        # if epoch % 5 == 0 or epoch == params.n_epochs - 1:
        #     predictions = inference(model, batch_sample_train, device=device)
        #     plot = \
        #         train_dataloader.dataset.get_predictions_plot(batch_sample_train,
        #                                                       predictions.cpu(),
        #                                                       batch_gt_train)
        #     writer.add_image('Predictions_train', plot,
        #                      epoch, dataformats='HWC')

        #     predictions = inference(model, batch_sample_val)
        #     plot = \
        #         train_dataloader.dataset.get_predictions_plot(batch_sample_val,
        #                                                       predictions.cpu(),
        #                                                       batch_gt_val)
        #     writer.add_image('Predictions_val', plot, epoch, dataformats='HWC')

        current_value = list(val_metrics.values())[0][0]
        is_best = current_value >= best_value

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_value = current_value
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                log_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': opt.state_dict(),
                               'scheduler_dict': lr_scheduler.state_dict(),
                               'best_value': best_value},
                               is_best=is_best,
                               ckpt_dir=ckpt_dir,
                               filename=ckpt_filename)

        logging.info("\ntrain loss: %.3f, val loss: %.3f" %
                     (train_loss, val_loss))
        for (train_metric_name, train_metric_results), \
            (val_metric_name, val_metric_results) in zip(train_metrics.items(),
                                                         val_metrics.items()):
            logging.info("train %s: %.3f, val %s: %.3f" % (train_metric_name,
                                                           train_metric_results[0],
                                                           val_metric_name,
                                                           val_metric_results[0]))

        logging.info("-"*20)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break
