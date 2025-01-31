import editdistance
import torch

from src.datautils.token_dataset_bressay import TOKEN_POSITION_CHAR_LEVEL_LIST, TOKEN_IS_CROSS_CHAR_LEVEL_LIST, \
    TOKEN_POSITION_CHAR_LEVEL_DICT, TOKEN_IS_READABLE_CHAR_LEVEL_LIST, EOS_STR_TOKEN, TOKEN_IS_CROSS_CHAR_LEVEL_DICT, \
    TOKEN_IS_READABLE_CHAR_LEVEL_DICT
from src.datautils.txt_transform import convert_int_to_chars, ctc_best_path_one, best_path
from src.evaluation.evaluate_reco import nb_chars_from_list, nb_words_from_list, edit_wer_from_list
from src.printutils.metrics_print import MetricLossCERWER, MetricLossCER


def evaluate_one_epoch(data_loader,
                       model,
                       device,
                       char_list,
                       token_blank,
                       eos_index):

    metrics_main_enc = MetricLossCERWER("Encoder Main")
    metrics_dec_txt = MetricLossCERWER("Decoder text")
    metric_final_dec = MetricLossCERWER("Decoder Final")

    # Tags predictions
    metrics_dec_pos_tag = MetricLossCER("Validation Decoder Position Tag")
    metrics_dec_cross_tag = MetricLossCER("Validation Decoder Cross Tag")
    metrics_dec_read_tag = MetricLossCER("Validation Decoder Readable Tag")

    model.eval()

    with torch.no_grad():
        for index_batch, batch_data in enumerate(data_loader):
            x = batch_data["imgs"].to(device)
            x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]

            y_enc = batch_data["label_ind_enc"].to(device)
            y_len_enc = batch_data["label_ind_enc_length"]

            y_dec = batch_data["label_ind_dec"].to(device)

            y_dec_postag = batch_data["ind_pos"].to(device)
            y_dec_crosstag = batch_data["ind_is_cross"].to(device)
            y_dec_readabletag = batch_data["ind_is_readable"].to(device)

            gt_posttag_str = [
                convert_int_to_chars(gt, TOKEN_POSITION_CHAR_LEVEL_LIST, break_on_eos=True, filter_sos=True) for gt in
                y_dec_postag]
            gt_crosstag_str = [
                convert_int_to_chars(gt, TOKEN_IS_CROSS_CHAR_LEVEL_LIST, break_on_eos=True, filter_sos=True) for gt in
                y_dec_crosstag]
            gt_readtag_str = [
                convert_int_to_chars(gt, TOKEN_IS_READABLE_CHAR_LEVEL_LIST, break_on_eos=True, filter_sos=True) for gt
                in y_dec_readabletag]

            y_gt_txt = batch_data["label_str_raw"]

            # Remove text padding
            y_gt_txt = [t.strip() for t in y_gt_txt]

            nb_item_batch = x.shape[0]

            dec_txt, dec_tag_pos, dec_tag_cross, dec_tag_readable, encoder_out = model.predict_4decoders(
                x,
                y_dec,
                y_dec_postag,
                y_dec_crosstag,
                y_dec_readabletag)

            # Compute the losses
            # Encoder
            encoder_outputs_main, encoder_outputs_shortcut = encoder_out
            encoder_outputs_main = torch.nn.functional.log_softmax(encoder_outputs_main, dim=-1)

            ctc_loss_main = model.ctc_loss_fn(encoder_outputs_main, y_enc, x_reduced_len, y_len_enc)
            # # Decoder -> cannot compute evaluation loss -> two sequences have different size

            # Compute CER
            # Encoder
            encoder_outputs_main, encoder_outputs_shortcut = encoder_out
            # Main
            encoder_outputs_main = torch.nn.functional.log_softmax(encoder_outputs_main, dim=-1)

            metrics_main_enc.add_loss(ctc_loss_main.item(), nb_item_batch)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            encoder_outputs_main = encoder_outputs_main.transpose(0, 1)

            top_main_enc = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                            enumerate(encoder_outputs_main)]
            predictions_text_main_enc = [ctc_best_path_one(p, char_list, token_blank) if p is not None else "" for p in
                                         top_main_enc]

            predictions_text_main_enc = [t.strip() for t in predictions_text_main_enc]  # Remove text padding

            cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_main_enc)]

            metrics_main_enc.add_cer(sum(cers), nb_chars_from_list(y_gt_txt))
            metrics_main_enc.add_wer(edit_wer_from_list(y_gt_txt, predictions_text_main_enc),
                                     nb_words_from_list(y_gt_txt))

            # Decoder
            decoder_outputs_cpu = dec_txt.cpu()
            pred_dec_txt_base = [best_path(l, char_list, eos_index) for l in decoder_outputs_cpu]

            # Remove text padding
            pred_dec_txt = [t.strip() for t in pred_dec_txt_base]

            final_pred = pred_dec_txt

            cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, pred_dec_txt)]

            metrics_dec_txt.add_cer(sum(cers), nb_chars_from_list(y_gt_txt))
            metrics_dec_txt.add_wer(edit_wer_from_list(y_gt_txt, pred_dec_txt), nb_words_from_list(y_gt_txt))

            # Position
            decoder_outputs_cpu_pos_tags = dec_tag_pos.cpu()
            pred_dec_postags = [
                best_path(l, TOKEN_POSITION_CHAR_LEVEL_LIST, TOKEN_POSITION_CHAR_LEVEL_DICT[EOS_STR_TOKEN]) for l in
                decoder_outputs_cpu_pos_tags]

            # Eval with space padding
            cers = [editdistance.eval(u, v) for u, v in zip(gt_posttag_str, pred_dec_postags)]
            metrics_dec_pos_tag.add_cer(sum(cers), nb_chars_from_list(gt_posttag_str))

            # Cross
            decoder_outputs_cpu_cross_tags = dec_tag_cross.cpu()
            pred_dec_crosstags = [
                best_path(l, TOKEN_IS_CROSS_CHAR_LEVEL_LIST, TOKEN_IS_CROSS_CHAR_LEVEL_DICT[EOS_STR_TOKEN]) for l in
                decoder_outputs_cpu_cross_tags]

            # Eval with space padding
            cers = [editdistance.eval(u, v) for u, v in zip(gt_crosstag_str, pred_dec_crosstags)]
            metrics_dec_cross_tag.add_cer(sum(cers), nb_chars_from_list(gt_crosstag_str))

            # Readable
            decoder_outputs_cpu_readable_tags = dec_tag_readable.cpu()
            pred_dec_readtags = [
                best_path(l, TOKEN_IS_READABLE_CHAR_LEVEL_LIST, TOKEN_IS_READABLE_CHAR_LEVEL_DICT[EOS_STR_TOKEN])
                for l in decoder_outputs_cpu_readable_tags]

            # Eval with space padding
            cers = [editdistance.eval(u, v) for u, v in zip(gt_readtag_str, pred_dec_readtags)]
            metrics_dec_read_tag.add_cer(sum(cers), nb_chars_from_list(gt_readtag_str))

            cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, final_pred)]

            metric_final_dec.add_cer(sum(cers), nb_chars_from_list(y_gt_txt))
            metric_final_dec.add_wer(edit_wer_from_list(y_gt_txt, final_pred), nb_words_from_list(y_gt_txt))

            # Print first batch prediction
            if index_batch == 0:
                nb_pred_to_print = min(6, nb_item_batch)
                # Text
                for i in range(nb_pred_to_print):
                    print("-----Ground truth all:-----")
                    print(y_gt_txt[i])

                    # print("-----Predictions encoder main:-----")
                    # print(predictions_text_main_enc[i])
                    print("-----Predictions decoder text:-----")
                    print(pred_dec_txt[i])

                    print("-----Ground truth position tag:-----")
                    print(gt_posttag_str[i])
                    print("-----Predictions decoder position tag:-----")
                    print(pred_dec_postags[i])
                    print("-----Ground truth cross tag:-----")
                    print(gt_crosstag_str[i])
                    print("-----Predictions decoder cross tag:-----")
                    print(pred_dec_crosstags[i])
                    print("-----Ground truth readable tag:-----")
                    print(gt_readtag_str[i])
                    print("-----Predictions decoder readable tag:-----")
                    print(pred_dec_readtags[i])

    metric_final_dec.normalize()
    metrics_main_enc.normalize()
    metrics_dec_txt.normalize()

    metrics_dec_pos_tag.normalize()
    metrics_dec_cross_tag.normalize()
    metrics_dec_read_tag.normalize()

    dict_result = {
        "metrics_main_enc": metrics_main_enc,
        "metric_final_dec": metric_final_dec,
        "metrics_dec_txt": metrics_dec_txt,

        "metrics_dec_pos_tag": metrics_dec_pos_tag,
        "metrics_dec_cross_tag": metrics_dec_cross_tag,
        "metrics_dec_read_tag": metrics_dec_read_tag,
    }

    return dict_result
