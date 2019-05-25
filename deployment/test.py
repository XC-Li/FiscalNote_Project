"""
Unit Tests for stance detection and stance classification
"""

import unittest
from stance_detection_classification import StanceDetectionAndClassification


class TestStanceClassification(unittest.TestCase):
    def test_sc_prediction(self):
        sdac = StanceDetectionAndClassification()
        positive_text = """Mr. GRAYSON . Mr. Speaker, these statements and findings are made in support of the “Catching Up to 
    1968 Act of 2013.” In determining that it is time to raise the minimum wage to $10.50 per hour and index it to 
    inflation, Congress makes the following findings: (1) Since 1968, the minimum wage has lost nearly one-third of 
    its value. Had it kept pace with inflation since then, the federal minimum wage would be $10.67 today. (2) Given 
    that the minimum wage has not kept pace with inflation, more than thirty million low-wage workers are making less 
    today than low-wage workers did 45 years ago in 1968. (3) As the cost of living increased in the past several 
    decades, the reduced purchasing power of the minimum wage has made it more difficult for low-wage workers to pay 
    for basic necessities such as housing, transportation, food, and healthcare. (I) Housing prices have nearly 
    doubled; the median value of owner-occupied properties has increased by about 80 percent between 1970 and 2009. (
    II) The cost of a gallon of motor vehicle gasoline has increased more than 60 percent from 1978 to 2012 according 
    to U.S. Energy Information data. (III) The average cost of health insurance premiums has skyrocketed. According 
    to U.S. Census figures, from 1990 to 2009, health insurance costs per capita have more than doubled, increasing 
    102 percent. The average annual cost of employer-sponsored family health insurance premiums increased 89 percent 
    from 1999 to 2011. Workers bore more of that load, with the average worker contribution toward employer-sponsored 
    health insurance increasing by 94 percent. On top of this, an increasing number of medical expenses and services 
    are not paid for by health insurance, resulting in dramatically increasing out-of-pocket expenses—27 percent from 
    1996 to 2009—for families. (IV) Since just 1994, the average cost for a family of four to provide food for the 
    family has increased about 10 percent, according to figures from the USDA’s monthly estimates of food plans. (4) 
    The current federal minimum wage of $7.25 per hour, $15,080 annually, does not even meet the U.S. Census Bureau’s 
    poverty threshold for a family of two or the Department of Health and Human Service’s poverty guidelines for a 
    family of two, both of which are above $15,000 per year. (5) Worker productivity has more than doubled since the 
    1960s, according to Bureau of Labor Statistics’ data, yet all that low-wage workers have received for their 
    effort is the declining value of the minimum wage. (6) The failure of Congress to make sure that the minimum wage 
    keeps pace with inflation has exacerbated income inequality in this country and placed the American Dream out of 
    reach for many hard-working low-wage workers in this country. At the same time that the minimum wage has lost 
    nearly one-third of its value, the average income of the top 1 percent of taxpayers has skyrocketed. The 
    threshold for a family’s annual income to be considered in the top 1 percent of taxpayers increased from about 
    $75,000 in 1968 to over $1 million in 2011. Adjusting for inflation, the annual income of the top one percent has 
    more than doubled in that time, increasing 110 percent. Just before the recent financial crisis, the incomes of 
    the top one percent had nearly tripled from 1968 to 2007, increasing by 196 percent. (7) The top 100 highest paid 
    CEOs all made over $15 million last year. The highest paid CEO made over $131 million in 2012, the equivalent of 
    almost $63,000 per hour—$10,000 more than the median annual household income in the United States. (8) Though the 
    United States economy has begun to recover from the recent financial crisis, the unemployment rate is still 7.7 
    percent and there still remain 28.6 million unemployed or underemployed. Raising the minimum wage would help 
    stimulate the economy and create jobs. (I) Raising the minimum wage to $10.50 per hour would give a raise to more 
    than 30 million workers, add a net increase of over $30 billion in economic activity, and create more than 140,
    000 new jobs. (II) According to a Chicago Federal Reserve study, for every dollar increase to the hourly pay of a 
    minimum wage worker, the result is $2,800 in new spending from that worker’s household over the year. (9) 
    Two-thirds of low-wage workers are employed by large, profitable corporations. (10) Many large, multi-national 
    corporations pay higher minimum wages in Canada and Europe, and still remain profitable. (11) Without raising the 
    minimum wage and indexing it to inflation, it becomes more likely E358 that low-wage workers will fall further 
    into poverty and be more reliant on government services like food stamps, Medicaid, welfare, and the earned 
    income tax credit. These government services are paid for by the taxpayers and other small businesses. In this 
    sense, many small businesses that already pay their employees more than the federal minimum wage end up 
    subsidizing the profitability of their large corporate competitors. This is a perversion of capitalism. Raising 
    the minimum wage would not put small businesses like this at a competitive disadvantage, but could in fact help 
    them. For instance, according to the MO Healthnet Employer Report, in Missouri during the first quarter of 2011 (
    the most recent data) the total cost to the state of the 50 employers whose employees rely most heavily on 
    Medicaid was about $43.5 million. According to data from the state Department of Job and Family Services, 
    the State of Ohio paid $111.5 million in 2007 for Medicaid costs for workers and their dependents at 50 employers 
    statewide. (12) Nearly two-thirds of minimum wage workers are women. A greater proportion of minimum wage workers 
    are black (15 percent) or Hispanic (20.2 percent) than of the population as a whole (13.1 percent black, 
    and 16.7 percent Hispanic). (13) The United States has one of the lowest minimum wages when compared with other 
    Western, industrialized countries. Australia’s minimum wage is more than double the minimum wage in the United 
    States—at about $16 per hour. Of ten countries with minimum wages higher than the United States’, eight of them 
    have unemployment rates lower than ours, based on the most recent data available. (14) Poll after poll has shown 
    that about 70 percent of the American public supports increasing the minimum wage. """
        self.assertGreater(sdac.stance_classification(positive_text), 0.7)
        negative_text = """Mr. VAN HOLLEN . Mr. Speaker, I rise in opposition to the so-called “Reducing Regulatory Burdens Act,” 
        which would roll back clean water protections and allow untracked pesticide pollution in our rivers and streams. More than two and a half years ago, 
        the Environmental Protection Agency put in place basic pesticide protections by requiring a general permit for the direct application of pesticides 
        into waterways. Nearly 2,000 waterways are already contaminated by pesticides, harming fish and amphibians and potentially accumulating in people who 
        eat those fish. The commonsense permit does not affect land applications of pesticides, maintains existing agricultural exemptions, and allows for 
        immediate spraying to protect the public from vector-borne diseases. Today’s legislation would roll back the permitting rule, leaving pesticide
         application unmonitored and our waterways vulnerable to contamination. I urge a “no” vote.
        """
        self.assertLess(sdac.stance_classification(negative_text), 0.3)

    def test_sd_prediction(self):
        sdac = StanceDetectionAndClassification()
        contain_stance = """Mr. GRAYSON . Mr. Speaker, these statements and findings are made in support of the “Catching Up to 
    1968 Act of 2013.” In determining that it is time to raise the minimum wage to $10.50 per hour and index it to 
    inflation, Congress makes the following findings: (1) Since 1968, the minimum wage has lost nearly one-third of 
    its value. Had it kept pace with inflation since then, the federal minimum wage would be $10.67 today. (2) Given 
    that the minimum wage has not kept pace with inflation, more than thirty million low-wage workers are making less 
    today than low-wage workers did 45 years ago in 1968. (3) As the cost of living increased in the past several 
    decades, the reduced purchasing power of the minimum wage has made it more difficult for low-wage workers to pay 
    for basic necessities such as housing, transportation, food, and healthcare. (I) Housing prices have nearly 
    doubled; the median value of owner-occupied properties has increased by about 80 percent between 1970 and 2009. (
    II) The cost of a gallon of motor vehicle gasoline has increased more than 60 percent from 1978 to 2012 according """
        self.assertGreater(sdac.stance_detection(contain_stance), 0.7)
        no_stance = """
        Mr. MICA . Mr. Speaker, I rise today to congratulate our ally and friend, the Republic of slovakia, on her 20th anniversary of independence. In two brief decades, Slovakia has dramatically transitioned to an independent, democratic and economically viable free nation. As some of my colleagues may know, my great grandparents emigrated from Slovakia to the United States at the turn of the last century. Like so many others, my family was drawn to America by the promises of freedom and opportunity. My ancestors would be proud to see both the progress of America over that century and the positive development of the Slovak Republic in its 20 years of independence. For a millennia, the Slovak people were ruled or governed by others. After centuries of power shifts and realignments, in 1989, the Velvet Revolution brought down the communist regime in Czechoslovakia. Democracy came to that nation as formerly jailed dissident and political activist Vaclav Havel was elected to the presidency. However, the Slovak people’s yearning for self-governance was not realized until 1993. Following the peaceful separation of the Czech and Slovak Republics, January 1, 1993 marks the birth of the Second Slovak Republic. As fate would have it, days later I was sworn in as a Member of the U.S. House of Representatives. As one of the Members of Congress with Slovak ancestry, I have been proud to work with many who have been so successful in strengthening U.S.-Slovak relations and to aid in the political and economic development of the Slovak Republic. Like any new democracy, the Slovak Republic has experienced some growing pains. After President Michal Kovács service as the first president, my good friend and former Kosice Mayor Rudolf Schuster was elected president after a constitutional amendment changed the presidency to a directly elected position. His successor is now President Ivan Gasparovic. I commend these and all the other Slovak leaders who have helped fashion a new era for their people. Even with many difficult challenges as a new nation, the Slovak Republic made outstanding progress over the last 20 years, and I am proud to have played a very small part in its history. In 2000, Slovakia became a member of the Organization for Economic Co-operation and Development and in 2004, joined both NATO and the European Union. The Republic of Slovakia and its people continue to provide international leadership both in Europe and throughout the world. For the United States and the American people, we are fortunate to have such a strong ally and friend in the family of nations. So today we salute and congratulate the Slovak Republic on the special occasion of their 20th anniversary of independence. We wish them every continued future success as they mark this historic milestone. I ask my colleagues to join me in congratulating the Slovak Republic and look forward to peace and prosperity for both of our countries for decades to come.
        """
        self.assertLess(sdac.stance_detection(no_stance), 0.3)

    def test_sd_short_input(self):
        sdac = StanceDetectionAndClassification()
        short_speech = "Short Speech"  # will be determined as not contain stance, return -1
        self.assertEqual(sdac.stance_detection(short_speech), -1)


if __name__ == '__main__':
    unittest.main()
