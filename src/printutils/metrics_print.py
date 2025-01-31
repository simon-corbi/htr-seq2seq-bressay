class MetricLossCER(object):
    """
    """
    def __init__(self, tag):
        self.tag = tag
        self.loss = 0
        self.nb_item_loss = 0

        self.cer = 0
        self.nb_letters = 0

    def add_cer(self, cer, nb_letters):
        self.cer += cer
        self.nb_letters += nb_letters

    def add_loss(self, loss, nb_item):
        self.loss += loss
        self.nb_item_loss += nb_item

    def normalize(self):
        if self.nb_letters != 0:
            self.cer /= self.nb_letters

        if self.nb_item_loss != 0:
            self.loss /= self.nb_item_loss

    def print_values(self):

        print_str = self.tag
        print_str += " : "
        print_str += f"Loss: {self.loss:.3f}; "
        print_str += f"CER: {100 * self.cer:.2f}%; "

        print(print_str)


class MetricLossCERWER(MetricLossCER):
    """
    """

    def __init__(self, tag):
        super(MetricLossCERWER, self).__init__(tag)

        self.wer = 0
        self.nb_words = 0

    def add_wer(self, wer, nb_words):
        self.wer += wer
        self.nb_words += nb_words

    def normalize(self):
        super().normalize()

        if self.nb_words != 0:
            self.wer /= self.nb_words

    def print_values(self):

        print_str = self.tag
        print_str += " : "
        print_str += f"Loss: {self.loss:.3f}; "
        print_str += f"CER: {100 * self.cer:.2f}%; "
        print_str += f"WER: {100 * self.wer:.2f}%; "

        print(print_str)

    def print_cer(self):

        print_str = self.tag
        print_str += " : "
        print_str += f"CER: {100 * self.cer:.2f}%; "

        print(print_str)
