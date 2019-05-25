"""By: Xiaochi (George) Li: github.com/XC-Li"""
import re


class RuleBasedStanceDetection(object):
    def __init__(self):
        # other positive keyword: call on|call for|important,[need,come] .* to pass|vote[d] [A,a]ye|cosponsor
        self.positive_detector = re.compile('support')
        self.negative_detector = re.compile('opposition|oppose')
        self.confused_detector = re.compile('passed|introduce|introduction|rise|will vote')
        self.contain_bill = re.compile('H[0-9]{4}|H.R.|[A,a]ct')
        self.pn_count = []

    # def stance_detector(self, speech):
    #     if self.positive_detector.search(speech):
    #         return 1
    #     elif self.negative_detector.search(speech):
    #         return -1
    #     elif self.confused_detector.search(speech):
    #         return 2
    #     elif self.contain_bill.search(speech):
    #         return 3
    #     else:
    #         return 0

    def stance_detector(self, speech, pn_ratio=1, cutoff=None):
        """
        predict by the count of positive/negative keyword
        Args:
            speech(str): the speech
            pn_ratio(int,float): the parameter to control the weight of negative keywords
            when both positive and negative keywords appear in the speech
            cutoff(int): the cutoff point of the speech, only detect the keyword before cutoff
        Returns:
            int: -1(negative),1(positive),0(not detected),2(confused),3(contain bill)
        """
        if cutoff:
            speech = speech[:cutoff]
        if self.positive_detector.search(speech):
            positive_count = len(self.positive_detector.findall(speech))
        else:
            positive_count = 0
        if self.negative_detector.search(speech):
            negative_count = len(self.negative_detector.findall(speech))
        else:
            negative_count = 0
        if positive_count == 0 and negative_count == 0:
            if self.confused_detector.search(speech):
                return 2
            if self.contain_bill.search(speech):
                return 3
            return 0
        else:
            self.pn_count.append([positive_count, negative_count])
            # print(positive_count, negative_count)
            if positive_count > pn_ratio * negative_count:
                return 1
            else:
                return -1

    def stance_detection_labeler(self, speech, strict=True, pn_ratio=1, cutoff=None):
        stance = self.stance_detector(speech, pn_ratio, cutoff)
        if strict:  # only relabel when certain
            if stance == 1 or stance == -1:
                return 1
            else:
                return -1
        else:  # relabel when possible
            if stance != 0:
                return 1
            else:
                return -1

    def stance_classification_labeler(self, speech, pn_ratio=1, cutoff=None):
        stance = self.stance_detector(speech, pn_ratio, cutoff)
        if stance == 1 or stance == -1:
            return stance
        else:
            return 0


