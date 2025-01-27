# from __future__ import print_function
import argparse
import torch.utils.data
from dataloader import get_dataloader
import Model
import os
# import Visulaizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--root_dir",
    #     default="",
    #     help="path to dataset",
    # )
    # parser.add_argument(
    #     "--train_iam_list",
    #     default="/home/media/TIP_Bina/data/binarization/train_pair_for_2011.lst",
    #     help="path to dataset",
    # )
    parser.add_argument(
        "--hdw-files", "-hdw", type=str, nargs="+", help="Hand written (dirty) files"
    )

    parser.add_argument(
        "--cad-files", "-cad", type=str, nargs="+", help="CAD (clean) files"
    )

    parser.add_argument(
        "--limit-mem",
        "-lm",
        action="store_true",
        help="Limit memory option.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=0,
        help="Size of patches to take on images (If > 0, overides --max-size).",
    )

    parser.add_argument(
        "--patch-step",
        type=int,
        default=-1,
        help="Step to jump from one patch to the other.",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=0,
        help="Size to reshape the inputs for training.",
    )

    parser.add_argument("--workers", default=8, help="number of data loading workers")
    parser.add_argument("--batchSize", type=int, default=16, help="input batch size")
    parser.add_argument(
        "--niter", "-n", type=int, default=1, help="number of epochs to train for"
    )

    parser.add_argument(
        "--niter_texture",
        "-nt",
        type=int,
        default=1,
        help="number of epochs to train for texture network",
    )

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--cuda", default="cuda", help="enables cuda")
    parser.add_argument(
        "--eval_freq_iter", type=int, default=1, help="Interval to be displayed"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.5, help="beta2 for adam. default=0.5"
    )
    parser.add_argument("--print_freq_iter", type=int, default=100)
    parser.add_argument(
        "--niter_decay",
        type=int,
        default=100,
        help="# of iter to linearly decay learning rate to zero",
    )
    parser.add_argument(
        "--norm_G",
        type=str,
        default="instance",
        help="instance normalization or batch normalization",
    )
    parser.add_argument(
        "--norm_D",
        type=str,
        default="instance",
        help="instance normalization or batch normalization",
    )
    parser.add_argument(
        "--out_chan", type=int, default=1, help="# of output image channels"
    )
    parser.add_argument(
        "--in_chan", type=int, default=1, help="Number of channels in the images"
    )

    parser.add_argument(
        "--gpu-id",
        "-gid",
        type=int,
        default=0,
        help="Set  gpu cuda device id.",
    )

    parser.add_argument(
        "--multi-gpu",
        "-mgpu",
        action="store_true",
        help="Enables multi-gpu operations.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force device to cpu.",
    )

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if opt.cpu:
        device = torch.device("cpu")
        print("Forcing device to cpu (%s)" % device.type)

    print("Using device %s" % device.type)
    if device.type == "cuda":
        print('"current" device id:', torch.cuda.current_device())
        print("GPU id #%d" % device.index)

    opt.device = device

    print("Number of cpu: %d" % os.cpu_count())
    opt.nb_proc = opt.workers  # os.cpu_count()  # // 2
    print("Number of cpu for the dataloader: %d" % opt.nb_proc)

    opt.isTrain = True
    train_loader = get_dataloader(opt)
    model = Model.Model(opt)
    model.to(device)
    tensorboard_viz = None  # Visulaizer.Visualizer()

    if not os.path.exists("./checkpoints"):
        os.mkdir("checkpoints")

    step = 0

    # me
    model.train()

    for epochs in range(opt.niter_texture):
        for i_batch, (clean_img, degraded_img) in enumerate(train_loader):
            step = step + 1
            clean_img = clean_img.to(device)
            degraded_img = degraded_img.to(device)
            gen_img = model.forward_texture(clean_img, degraded_img)
            errors = model.get_current_errors_texture()
            if (step + 1) % opt.print_freq_iter == 0:
                img = model.img_list_after_texture()
                if tensorboard_viz:
                    tensorboard_viz.vis_image(img, step)
                print(
                    "Epoch: {}, Iter: {}, Steps: {}, Loss:{}".format(
                        epochs, i_batch, step, errors
                    )
                )
            print(
                "Epoch: {}, Iter: {}, Steps: {}, Loss:{}".format(
                    epochs, i_batch, step, errors
                )
            )
        torch.save(
            model.state_dict(),
            "./checkpoints/checkpoint_texture_epoch_{}.pth".format(epochs),
        )
    # torch.jit.save(torch.jit.trace(model.forward_texture, clean_img), "texture_fwd.jit")

    # model.load_state_dict(torch.load('./model/checkpoint_texture_epoch_14.pth'))
    print("ok1")
    model.freeze_network(model.Texture_generator)
    model.freeze_network(model.Texture_Discrimator)

    steps = 0
    for epoch in range(opt.niter):
        for i_batch_bin, (clean_img, degraded_img) in enumerate(train_loader):
            steps = steps + 1
            clean_img = clean_img.to(device)
            degraded_img = degraded_img.to(device)
            gen_img = model.Texture_generator(clean_img, degraded_img)
            gen_img_bin = model.forward_binaziation(gen_img, clean_img, degraded_img)
            error_bin = model.get_current_errors_bin()
            print(
                "Epoch: {}, Iter: {}, Steps: {}, Loss:{}".format(
                    epoch, i_batch_bin, step, error_bin
                )
            )

            if (steps + 1) % opt.print_freq_iter == 0:
                img = model.img_list_after_bin()
                if tensorboard_viz:
                    tensorboard_viz.vis_image(img, step)
                print(
                    "Epoch: {}, Iter: {}, Steps: {}, Loss:{}".format(
                        epoch, i_batch_bin, step, error_bin
                    )
                )

        torch.save(
            model.state_dict(),
            "./checkpoints/checkpoint_binarization_epoch_{}.pth".format(epoch),
        )
    # torch.jit.save(torch.jit.trace(model, clean_img), "texture_bin.jit")

    model.Un_freeze_network(model.Texture_generator)
    model.Un_freeze_network(model.Texture_Discrimator)
    print("ok2")
    torch.autograd.set_detect_anomaly(True)

    steps = 0
    for epoch in range(opt.niter):
        for i_batch_bin, (clean_img, degraded_img) in enumerate(train_loader):
            steps = steps + 1
            clean_img = clean_img.to(device)
            degraded_img = degraded_img.to(device)
            print("Entering joint forward")
            model.joint_forward(clean_img, degraded_img)
            error_bin = model.get_current_errors_bin()

            if (step + 1) % opt.print_freq_iter == 0:
                img = model.img_list_after_texture()
                if tensorboard_viz:
                    tensorboard_viz.vis_image(img, step)
                print(
                    "Epoch: {}, Iter: {}, Steps: {}, Loss:{}".format(
                        epoch, i_batch_bin, step, error_bin
                    )
                )

        torch.save(
            model.state_dict(),
            "./checkpoints/checkpoint_joint_epoch_{}.pth".format(epoch),
        )

#     # on the fly hacking of the function
#     model_utils.eval_1epoch = eval_1epoch
#     model_utils.train_1epoch = train_1epoch

#     model_utils.train(
#         model,
#         train_dloader,
#         loss_fn,
#         optimizer,
#         N_epochs=args.n_epochs,
#         scheduler=scheduler,
#         scheduler_per_batch=BATCH_SCHEDULER,
#         eval_dloader=eval_dloader,
#         score_fn=score_fn,
#         pgbar=args.progress_bar,
#         model_name=model_name + ".pth",
#         early_stop=EARLY_STOP,
#         verbose=1,
#         device=device,
#         epoch_start=EPOCH_START,
#         restart_dict=RESTART_DICT,
#         jit_format=True,
#         input_shape=input_shape,
#     )

#     return


# def train_1epoch(
#     model,
#     dloader,
#     loss_fn,
#     optimizer,
#     scheduler=None,
#     scheduler_per_batch=False,
#     score_fn=None,
#     pgbar=True,
#     lr_curve=None,
#     device="cpu",
# ):
#     """Train one epoch of the model"""
#     # current value of the learning rate (unchanged if no scheduler)
#     lr_epoch = optimizer.param_groups[0]["lr"]
#     # model in train mode
#     model.train()
#     epoch_loss = 0
#     # score associated to the training data evaluated with score_fn for each epoch
#     epoch_score = 0
#     # loop over each batch of (inputs, targets)
#     for inputs, targets, masks in (
#         alive_it(dloader, bar="bubbles", title="Training:") if pgbar else dloader
#     ):
#         # move batch to device (TO DO: directly in the dloader, see also test_dloader)
#         inputs, labels, masks = inputs.to(device), targets.to(device), masks.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass of the model
#         outputs = model(inputs)
#         # the output could be logits and the labels can be standard normed

#         # compute the loss
#         loss = loss_fn(outputs * (1.0 - masks), labels * (1.0 - masks))

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         # Compute running loss average of mini-batch
#         epoch_loss += loss.item() * inputs.size(0)
#         # if score_fn computes the score of the prediction wrt the targets for each batch
#         if score_fn:
#             epoch_score += score_fn(
#                 outputs * (1.0 - masks), labels * (1.0 - masks)
#             ) * inputs.size(0)

#         # if a scheduler per batch, update the learning rate for next batch and lr_curve
#         if scheduler and scheduler_per_batch:
#             lr_epoch = scheduler.get_last_lr()[0]
#             lr_curve.append(lr_epoch)
#             # forward step of scheduler
#             scheduler.step()

#     # if a scheduler per epoch, update learning rate for next epoch
#     if scheduler and not scheduler_per_batch:
#         lr_epoch = scheduler.get_last_lr()[0]
#         lr_curve.append(lr_epoch)
#         # forward step of scheduler
#         scheduler.step()

#     # end loop over batch for this epoch: computes the loss of that epoch
#     epoch_loss = epoch_loss / len(dloader.dataset)
#     if score_fn:
#         epoch_score = epoch_score / len(dloader.dataset)

#     return epoch_loss, epoch_score, lr_epoch


# @torch.no_grad
# def eval_1epoch(
#     model,
#     dloader,
#     loss_fn,
#     score_fn=None,
#     pgbar=True,
#     device="cpu",
# ):
#     model.eval()
#     # loss for each epoch of the evaluation set if eval_dloader
#     epoch_loss = 0
#     # score associated to the evaluation set if eval_dloader for each epoch
#     epoch_score = 0
#     # score_mssim = 0
#     for inputs, targets, masks in (
#         alive_it(dloader, bar="brackets", title="Evaluation:") if pgbar else dloader
#     ):
#         inputs, labels, masks = inputs.to(device), targets.to(device), masks.to(device)
#         outputs = model(labels)  # inputs

#         loss = loss_fn(outputs * (1.0 - masks), labels * (1.0 - masks))
#         epoch_loss += loss.item() * inputs.size(0)
#         if score_fn:
#             epoch_score += score_fn(
#                 outputs * (1 - masks), labels * (1 - masks)
#             ) * inputs.size(0)
#         # score_mssim += mssim(outputs * (1 - masks), labels * (1 - masks)) * inputs.size(
#         #     0
#         # )

#     # compute the loss and score evaluation for that epoch
#     epoch_loss = epoch_loss / len(dloader.dataset)
#     if score_fn:
#         epoch_score = epoch_score / len(dloader.dataset)
#     return epoch_loss, epoch_score
