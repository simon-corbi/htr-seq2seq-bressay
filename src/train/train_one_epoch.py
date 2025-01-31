def train_one_epoch(training_loader,
                    optimizer,
                    model,
                    device,
                    p_teacher_forcing):
    loss_enc_main_epoch = 0

    loss_dec_text_epoch = 0

    loss_dec_pos_epoch = 0
    loss_dec_cross_epoch = 0
    loss_dec_readable_epoch = 0

    total_loss_e = 0

    model.train()

    for index_batch, batch_data in enumerate(training_loader):
        optimizer.zero_grad()

        x = batch_data["imgs"].to(device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]

        y_enc = batch_data["label_ind_enc"].to(device)
        y_len_enc = batch_data["label_ind_enc_length"]

        y_dec = batch_data["label_ind_dec"].to(device)

        y_dec_postag = batch_data["ind_pos"].to(device)
        y_dec_crosstag = batch_data["ind_is_cross"].to(device)
        y_dec_readabletag = batch_data["ind_is_readable"].to(device)

        dec_txt, dec_tag_pos, dec_tag_cross, dec_tag_readable, encoder_out, \
        total_loss, ctc_loss_main, loss_dec_text, loss_dec_pos_tag, loss_dec_cross_tag, loss_dec_read_tag = model.forward_4decoders(
            x,
            y_enc,
            x_reduced_len,
            y_len_enc,
            y_dec,
            y_dec_postag,
            y_dec_crosstag,
            y_dec_readabletag,
            p_teacher_forcing=p_teacher_forcing,
            use_teacher_forcing=True)

        total_loss.backward()

        optimizer.step()

        total_loss_e += total_loss.item()

        loss_enc_main_epoch += ctc_loss_main.item()
        loss_dec_text_epoch += loss_dec_text.item()
        loss_dec_pos_epoch += loss_dec_pos_tag.item()
        loss_dec_cross_epoch += loss_dec_cross_tag.item()
        loss_dec_readable_epoch += loss_dec_read_tag.item()

    total_loss_e /= (index_batch + 1)
    loss_enc_main_epoch /= (index_batch + 1)

    loss_dec_text_epoch /= (index_batch + 1)
    loss_dec_pos_epoch /= (index_batch + 1)
    loss_dec_cross_epoch /= (index_batch + 1)
    loss_dec_readable_epoch /= (index_batch + 1)

    losses = {
        "total_loss_e": total_loss_e,
        "loss_enc_main_epoch": loss_enc_main_epoch,

        "loss_dec_text_epoch": loss_dec_text_epoch,

        "loss_dec_pos_epoch": loss_dec_pos_epoch,
        "loss_dec_cross_epoch": loss_dec_cross_epoch,
        "loss_dec_readable_epoch": loss_dec_readable_epoch
    }
    return losses
