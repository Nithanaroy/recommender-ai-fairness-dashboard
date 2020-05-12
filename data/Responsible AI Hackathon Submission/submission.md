<span class="c0"></span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 4.00px;">![](images/image14.png "horizontal line")</span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 6.67px;">![](images/image15.png "horizontal line")</span><span
class="c82 c14 c92">                                                   
                          </span>

<span class="c14 c33">Responsible AI Hackathon</span>

<span class="c0"></span>

<span class="c73">I</span><span class="c34">mpact of user personality on
advertisement recommendations</span>

<span class="c59 c62 c89"></span>

<span class="c7">Responsible AI Concerns</span>
===============================================

<span class="c0">Bias and discrimination in marketing advertisements are
something big firms and governments have been tackling for years. With
the advent of artificial intelligence and a gigantic amount of data
about customers (personal information, interactions, likes, dislikes )
and products,  we have machine learning algorithms and recommendation
systems to automate audience targeting, ad delivery, and ad engagement
predictions. But little did they know that bias in the data and opacity
of the models would land them into trouble for the same problem - BIAS &
DISCRIMINATION</span>

<span class="c0"></span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 449.33px;">![](images/image16.png)</span>

<span class="c0"></span>

<span class="c0"></span>

<span class="c59 c77 c62">❝</span>

<span class="c62 c56 c72">Women control 73 percent of consumer spending
in the United States and $20 trillion globally and yet ads frequently
fail to speak to them in a way that shows an understanding of their
lives.</span>

<span class="c77">❞</span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 462.67px;">![](images/image11.png)</span>

<span class="c0"></span>

<span class="c0">Based on our study on existing problems, we found the
following broad areas, each one being a research area of its own kind,
where the recommended advertisement can be deemed as biased or
discriminatory and not well accepted by society. </span>

<span class="c42 c14">Bias in Online ad delivery & audience
targeting</span>

<span class="c0">Big firms nowadays target their customers using click
prediction models and recommendation algorithms to drive their profits.
However, due to inherent bias in the data that has been collected over
the years, the delivery of ads can be biased.</span>

<span
class="c1"><a href="https://www.google.com/url?q=https://www.theverge.com/2019/4/4/18295190/facebook-ad-delivery-housing-job-race-gender-bias-study-northeastern-upturn&amp;sa=D&amp;ust=1589252851987000" class="c15">https://www.theverge.com/2019/4/4/18295190/facebook-ad-delivery-housing-job-race-gender-bias-study-northeastern-upturn</a></span>

<span class="c14 c42">Advertisement not reaching everyone</span>

<span class="c0">Although it's prerogative of the companies to decide
the audience and target customers who deliver high profit, sometimes it
can kick in sense of discrimination across the customers who learn about
the product indirectly say by word of mouth. For example, beauty
products being excessively targeted for only white women.</span>

<span class="c42 c14">Advertisements delivered to the right audience but
offending and harassing customers</span>

<span class="c0">Over relying on machine learning models might keep
recommending and targeting a certain section of the population without
taking into effect the perception and acceptance of the ads.</span>

<span class="c0">For example, advertisements for sports goods delivered
to the members of the black community without taking into account the
profession or interests of the person. On similar lines, we see ads
delivered to our mailboxes for some disease/ailments which one wouldn't
want to discuss.</span>

<span class="c42 c14">Bias in the advertisement images</span>

<span class="c0">With all the above boxes checked correctly, we could
still have some possibility of bias or discrimination via images that
the ads contain.</span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 260.50px; height: 319.30px;">![](images/image6.png)</span>

<span class="c59 c14 c71 c91">The image showing women showing success
linked to kitchen and microwave</span>

<span class="c40 c14">Source - </span><span
class="c1 c57 c14"><a href="https://www.google.com/url?q=https://in.pinterest.com/kminseo63/bias-in-advertisements/&amp;sa=D&amp;ust=1589252851988000" class="c15">https://in.pinterest.com/kminseo63/bias-in-advertisements/</a></span>

<span class="c7">The problem addressed and dataset</span>
=========================================================

<span class="c0">In this big world of online advertisements, there is
tons of research published in the field of machine learning bringing
profits for big firms, case studies on how well advertisements are being
accepted, and experiments linking psychology, politics with
advertisements.</span>

<span class="c0">We have picked one of the research studies linking user
personality with advertisements and setting a benchmark for ad rating
predictions and ad click predictions.</span>

<span class="c33 c14">Choosing a Dataset</span>
-----------------------------------------------

<span class="c0">For the task of investigating and fixing fairness in an
Advertisement Recommendation system requires a dataset that is rich
across 3 different verticals:</span>

1.  <span class="c0">The dataset should contain information about the
    people being presented with the Ads</span>
2.  <span class="c0">The dataset should have Advertisements across
    multiple categories and types</span>
3.  <span class="c0">The dataset should capture users' reactions and/or
    preferences for the Ads shown.</span>

With these constraints in mind, we have chosen to use the publicly
available ADS Dataset from research study: <span
class="c1"><a href="https://www.google.com/url?q=https://www.kaggle.com/groffo/ads16-dataset&amp;sa=D&amp;ust=1589252851990000" class="c15">https://www.kaggle.com/groffo/ads16-dataset</a></span> (Research
paper <span
class="c1"><a href="https://www.google.com/url?q=http://ceur-ws.org/Vol-1680/paper3.pdf&amp;sa=D&amp;ust=1589252851990000" class="c15">here</a></span>)

<span class="c0">This research uses a personality perspective to
determine the unique associations among the consumer's buying tendency
and advert recommendations.</span>

<span class="c33 c14">Why this Dataset?</span>
----------------------------------------------

<span class="c0">Personality-based Ads recommender systems are
increasingly attracting the attention of researchers and industry
practitioners. Personality is the latent construct that accounts for
“individuals characteristic patterns of thought, emotion, and
behavior.</span>

<span class="c0">Attitudes, perceptions, and motivations are not
directly apparent from clicks on advertisements or online purchases, but
they are an important part of the success or failure of online marketing
strategies. As a result, companies are increasingly tuning their Ads
recommendation systems upon personality factors. </span>

<span class="c0">We believe that "inferring a personality-based
recommendation" is an area where even a small amount of bias and a lack
of fairness can have a profound impact - not only to the consumers being
presented only with a selective set of Ads but also on the merchants
missing out on potential buyers by not targeting their ads
fairly.</span>

<span class="c33 c14">Dataset At A Glance</span>
------------------------------------------------

1.  <span class="c0">Information about anonymous 120 users. Multiple
    dimensions about demographic info (Age, Gender, Country, ZipCode
    etc.) and a multitude of personality indicating information like
    Most Listened  Music, Movies, Most visited Websites, Favorite Sports
    etc Additionally each user was asked to submit 10 images that they
    consider as "positive" (for example cat images) and 10 images they
    consider "negative" (for example an image showing a disagreement
    between people).</span>
2.  <span class="c0">300 Advertisements - categorized into 20 sections -
    ranging from Electronics, Automobiles etc to Kitchen, Pet supplies,
    sports supplies etc. Each category has 15 advertisements.</span>
3.  <span class="c0">Ratings provided by each user to each Advertisement
    on a scale of 1 - 5, highly likable getting a high rating.</span>
4.  <span class="c0">As seen from the point "3" above, the dataset
    captures exhaustively how all users rated all Ads - which makes this
    dataset an ideal candidate to build a model and analyze the fairness
    and/or bias - across various dimensions.</span>

<span class="c7">What we did</span>
===================================

### <span class="c38 c14">Data preparation</span>

<span class="c0">The ADS-16 dataset contains both structured and
unstructured data. So we first built a tailored dataset as explained in
Figure 1</span>

### <span class="c43"></span>

<span id="t.fc77d7d6d792b398c2675fa6da222fe7d334f5cd"></span><span
id="t.0"></span>

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><h4 id="h.ynge19vaqhqc" class="c67"><span class="c69 c14">Structured data </span></h4></td>
<td><h4 id="h.efvbrg5qiv3" class="c67"><span class="c14 c69">Unstructured data</span></h4></td>
</tr>
<tr class="even">
<td><ul>
<li><span class="c0">User preferences &amp; personal information- 120 users</span></li>
<li><span class="c0">Ratings for each ad shown to them</span></li>
</ul>
<p><span class="c0">300 ads per user</span></p></td>
<td><ul>
<li><span class="c0"> Advertisement images - 15 per category</span></li>
<li><span class="c0">Positive and Negative images - 10 per user</span></li>
</ul></td>
</tr>
</tbody>
</table>

#### <span class="c78 c62 c64"></span>

#### <span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 663.93px; height: 462.83px;">![](images/image8.png)</span>

<span class="c55">        Figure 1. Creating structured dataset by
combining information from resources in ADS-16 </span>

Google vision API was used on Ad images to extract rich semantic
information from them. Several encoding techniques like multi label
binarizer, one-hot encoding and Glove embeddings are used to transform
categorical data into numbers. We experimented with multiple word
embeddings and picked the one which gave the best based AUC score. The
final encoded training data is available <span
class="c1"><a href="https://www.google.com/url?q=https://github.com/salilkanitkar/responsible_ai_hackathon/blob/master/dataset/users-ads-without-gcp-ratings_OHE_MLB_FAV_UNFAV_Merged.csv&amp;sa=D&amp;ust=1589252851993000" class="c15">here</a></span>.

### <span class="c14 c64">Model Architecture </span>

We then trained a neural network model using Keras Functional API as a
classification problem where the model is tasked to predict the rating
for a given User and Ad combination. The high level model architecture
is shown in Figure 2.

### <span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 406.67px;">![](images/image2.jpg)</span>

<span class="c71">         </span><span class="c71">Figure 2
</span><span class="c26">Model architecture using Glove embeddings and
dense layers with 20 & 10 units</span>

<span class="c14 c40">Source - </span><span
class="c1 c14 c57"><a href="https://www.google.com/url?q=https://www.kaggle.com/colinmorris/embedding-layers&amp;sa=D&amp;ust=1589252851994000" class="c15">https://www.kaggle.com/colinmorris/embedding-layers</a></span>

<span class="c26"></span>

We experimented with various feature combinations and hyperparameters
(HP) to identify the best model with validation AUC. Tensorboard’s HP
Params dashboard’s parallel coordinates plot was quite helpful to narrow
down the best HP combination. The notebook with data preparation and
model training source code is available <span
class="c1"><a href="https://www.google.com/url?q=https://github.com/salilkanitkar/responsible_ai_hackathon/blob/master/models/basic-model/nn-model.ipynb&amp;sa=D&amp;ust=1589252851994000" class="c15">here</a></span>.
For others to be able to reproduce our results here are our <span
class="c1"><a href="https://www.google.com/url?q=https://docs.google.com/spreadsheets/d/1v-nYiDA3elM1UP9stkB42MK0bTbuLxYJE7qAYDP8FHw/edit%23gid%3D925421130&amp;sa=D&amp;ust=1589252851994000" class="c15">training notes</a></span> and
<span
class="c1"><a href="https://www.google.com/url?q=https://tensorboard.dev/experiment/fkAOs09DSOKc52nhOtO1XA/&amp;sa=D&amp;ust=1589252851995000" class="c15">tensoboard.dev links</a></span>.

<span class="c44">Fairness Metrics</span>
=========================================

<span class="c0">Before we can start assessment of potential unfairness
and bias, we need some methodology to measure it. The selection and
weighting of metrics is solely dependent on domain and our perception of
bias in society. For example, for gender bias one would care about fair
representation of both genders while in case of crime prediction one
would care about low misrepresentation rate. The following are the
standard metrics followed in the industry which is what we have used to
evaluate our models,</span>

### <span class="c22">False Positive Rate (FPR)</span>

<span class="c85">The false positive rate is calculated as the ratio
between the number of negative events wrongly categorized as positive
(false positives) and the total number of actual negative
events</span><span class="c87 c69 c62 c98 c91 c100"> </span>

<span class="c0">For example, in study for to classify toxic comments,
</span>

FPR (Religion:Christainity) = 0.16 and FPR (Religion:Muslim)=0.80,
clearly shows a bias based on religion.

### <span class="c22">Equal Opportunity Difference (EOD)</span>

<span class="c68 c62">This metric is computed as the difference of true
positive rates between the unprivileged and the privileged groups. The
true positive rate is the ratio of true positives to the total number of
actual positives for a given group.</span>

The ideal value is 0. A value of \< 0 implies higher benefit for the
privileged group and a value \> 0 implies higher benefit for the
unprivileged group.The definition of privileged and unprivileged depends
on hypotheses decided by domain experts.

### <span class="c22">Average Odds Difference (AOD)</span>

<span class="c62 c68">Computed as average difference of false positive
rate (false positives / negatives) and true positive rate (true
positives / positives) between unprivileged and privileged
groups.</span>

<span class="c0">The ideal value of this metric is 0. A value of \< 0
implies higher benefit for the privileged group and a value \> 0 implies
higher benefit for the unprivileged group. Fairness for this metric is
between -0.1 and 0.1</span>

<span class="c44">Bias Detection</span>
=======================================

### <span class="c82 c14 c64">Baseline Model results</span>

<span class="c0">As explained previously, we built a DNN Model using the
ADS16 dataset and analyzed how it performed across the Fairness Metrics
mentioned in the above section - the False Positive Rate (FPR), Equal
Opportunity Difference (EOD) and Average Odds Difference (AOD). These
provided us with two-fold advantages:</span>

1.  <span class="c0">We could see the bias and unfairness that our
    trained model showcased.</span>
2.  <span class="c0">We could attack these specific bias(es) with
    targeted mitigation strategies and evaluate if and how effective
    they are in reducing the bias. </span>

<span class="c0">In this section, let’s take a look at how our Baseline
Model performed. In subsequent sections, we will explain the mitigation
strategies applied.</span>

<span class="c0">Across the entire dataset - which includes
Advertisements from 20 different categories, we found two dimensions -
the Gender and the Age showcasing bias in the trained model. At a glance
some of these biases across the 3 Fairness Metrics looked like
below:</span>

-   <span class="c0">False Positive Rate for Females was 60% higher than
    Males (0.164 vs 0.09)</span>
-   <span class="c0">Equal Opportunity Difference was found to be 0.019.
    A greater than zero value here indicates higher benefit for the
    unprivileged group - which in our experiment is Male.</span>
-   <span class="c0">The False Positive Rate for the Young age group
    (less than 20 years) is 50% higher as compared to the Middle age
    group (between 20 to 40). 0.204 for Young vs 0.134 for Middle
    Age.</span>

<span class="c0">However, a more stark unfairness begins to show if
instead of looking across all Advertisement categories, we zoom in and
focus on a couple of particular categories of Ads. </span>

-   <span class="c53 c14">Age Bias in serving Sports Ads</span>

<!-- -->

-   <span class="c0">For the purpose of our analysis, we divided users
    into 3 age buckets. Age less than 20 years as “Young”, age between
    20 to 40 as “Middle Aged” and greater than 40 years as “Old”.</span>
-   <span class="c0">Seen below side-by-side are the two graphs showing
    the number of samples in each bucket on the left and the False
    Positive Rate for all buckets on the right. High FPR for the Young
    bucket.</span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 305.00px; height: 243.50px;">![](images/image7.png)</span><span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 307.00px; height: 239.17px;">![](images/image12.png)</span>

-   <span class="c0">Looking at the Equal Opportunity Difference between
    Young-MiddleAges and Young-Old buckets (-0.19 vs -0.17), it's clear
    that Ads are served mostly to the younger age group as the value of
    EOD is more negative.</span>
-   <span class="c0">We will explain how we mitigated this in the Bias
    Mitigation Section below.</span>

<span class="c0"></span>

-   <span class="c14">Gender Bias in Serving Consumer Electronics
    Ads</span>

<!-- -->

-   <span class="c0">We could also see a clear bias in serving Consumer
    Electronics Ads to Females and Males.</span>
-   <span class="c0">The EOD value is negative (-0.046) indicating that
    the bias is towards serving this to Male group.</span>
-   <span class="c0">The AOD value is more negative as well - indicating
    higher bias towards Male group. </span>
-   We explain how this was mitigated in the below section.

<span class="c44">Bias Mitigation</span><span class="c13"> </span>
==================================================================

### <span class="c14 c64">Approaches to address concerns - Bias Mitigation approaches </span>

1.  <span class="c43">Mitigation Approach 1 : Class balancing </span>

-   Oversampling : <span class="c14">S</span>ynthetic <span
    class="c14">M</span>inority <span class="c14">O</span>versampling
    <span class="c14">TE</span>chnique (<span
    class="c1"><a href="https://www.google.com/url?q=https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html&amp;sa=D&amp;ust=1589252851999000" class="c15">SMOTE</a></span><span
    class="c0">)</span>

<span class="c0">Classification using class-imbalanced data is biased in
favor of the majority class. The bias is even larger for
high-dimensional data, where the number of variables greatly exceeds the
number of samples. SMOTE is a data augmentation technique using which
new samples can be synthesized from the existing samples. </span>

<span class="c0">References : </span>

-   <span
    class="c1"><a href="https://www.google.com/url?q=https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html%23cbhk2002&amp;sa=D&amp;ust=1589252852000000" class="c15">imblearn - oversampling</a>
    </span>

<span class="c0"></span>

1.  <span class="c43">Mitigation Approach 2 : Reweighing </span>

<span class="c43"></span>

-    Preprocessing optimization - Reweighing using <span
    class="c1"><a href="https://www.google.com/url?q=https://aif360.mybluemix.net/&amp;sa=D&amp;ust=1589252852000000" class="c15">ai fairness 360</a></span><span
    class="c0"> </span>

<span class="c0"></span>

<span class="c0">Reweighing is a preprocessing technique that Weights
the examples in each (group, label) combination differently to ensure
fairness before classification . This modifies the weight of each
training example depending on whether the sample lies in privileged or
unprivileged class . </span>

<span class="c0"></span>

<span class="c0">References  : </span>

-   <span
    class="c1"><a href="https://www.google.com/url?q=https://link.springer.com/article/10.1007/s10115-011-0463-8&amp;sa=D&amp;ust=1589252852001000" class="c15">Data preprocessing techniques for classification without discrimination</a></span><span
    class="c0"> </span>
-   <span
    class="c1"><a href="https://www.google.com/url?q=https://arxiv.org/pdf/1810.01943.pdf&amp;sa=D&amp;ust=1589252852001000" class="c15">AI Fairness</a></span><span
    class="c1"><a href="https://www.google.com/url?q=https://arxiv.org/pdf/1810.01943.pdf&amp;sa=D&amp;ust=1589252852002000" class="c15"> </a></span><span
    class="c1"><a href="https://www.google.com/url?q=https://arxiv.org/pdf/1810.01943.pdf&amp;sa=D&amp;ust=1589252852002000" class="c15">360</a>
    </span>

<span class="c0"></span>

<span class="c0"></span>

### <span class="c14 c38">Effect of Bias Mitigation steps</span>

<span class="c0"></span>

-   <span class="c59 c75 c14">    Age Bias in serving sports Ads</span>

<span class="c0"></span>

<span class="c0">    We identified a bias in how the Sport Ads were
served to different age groups . </span>

<span class="c0"></span>

<span class="c0">    Age Groups :</span>

<span class="c0">        young : \< 20 yrs</span>

<span class="c0">        middleAged : 20 - 40 yrs</span>

<span class="c0">        old : \>40 yrs</span>

<span class="c0"></span>

<span class="c0">     </span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 385.33px;">![](images/image9.png "Chart")</span>

<span class="c0"></span>

<span class="c0">On Equal Opportunity Difference (EOD) comparison
between young-middleAged and young-old age groups . It's very clear that
Ads are served mostly to the younger age group as the values of EOD tend
to be more negative .  </span>

<span class="c0"></span>

<span class="c0">By using the mitigation steps , we can see that the EOD
improves (closer to 0). Both class balancing and reweighing techniques
help with EOD . </span>

<span class="c0"></span>

<span class="c0"></span>

<span class="c0"></span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 385.33px;">![](images/image13.png "Chart")</span>

<span class="c0"></span>

<span class="c0">Even with Average Odds Difference (AOD) , we see that
before mitigation steps were added , the values are more negative
indicating a bias in serving these ads to the younger population . With
mitigation , we improve the AOD (closer to 0) . We found the class
balancing brought a higher improvement compared to reweighting . </span>

<span class="c0"></span>

<span class="c0"></span>

<span class="c0"></span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 376.50px;">![](images/image4.png "Chart")</span>

<span class="c0">We also found that optimizing the model for fairness
increased the false positive rate. This is because adding these
mitigation steps diverts the objective of the model from only accuracy
to both accuracy and fairness.</span>

<span class="c0"></span>

<span class="c0"></span>

<span class="c0"></span>

<span class="c0"></span>

-   <span class="c59 c14 c75">  Gender Bias in serving consumer
    electronics Ads</span>

<span class="c0"></span>

<span class="c0"></span>

<span class="c0">The other bias we identified was in the serving of
consumer electronics ads between male and female groups .</span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 612.00px; height: 418.50px;">![](images/image5.png "Chart")</span>

<span class="c0"></span>

<span class="c0"></span>

In the above chart we see that , before any mitigation steps , the EOD
is negative indicating that the bias is towards serving this to male
group . Both the mitigation steps try to eliminate this by moving EOD
closer to zero . We see that both the mitigation steps are aggressive
and push EOD to positive value makes the model more biased towards
females . Even though re-weighing makes it biased towards female group
,the model overall is less biased with it (0.0287 more closer to zero
compared to -0.0463)  <span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 385.33px;">![](images/image3.png "Chart")</span>

<span class="c0"></span>

<span class="c0">With AOD , we see that both the mitigation steps make
the model less biased . Even in this case class balancing is more
aggressive . </span>

<span class="c0"></span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 385.33px;">![](images/image1.png "Chart")</span>

<span class="c0">Like in the previous case (Sport Ads) , we see that the
false positive rate increases with the introduction of the bias
mitigation steps .</span>

<span class="c0"></span>

### <span class="c38 c14">Recommendation</span>

<span class="c0"></span>

In the above cases we see that class balancing using SMOTE and
reweighing are effective in making the model less biased .  It's also
important to be aware of the impact of adding bias mitigation steps on
the model metrics like accuracy , auc etc . These fairness strategies
might have a negative effect on accuracy and a proper trade off must be
made and this should be dependent on model objective .

<span class="c7">Other explored Approaches</span>
=================================================

#### <span class="c23">Bias Metrics</span>

-   <span
    class="c1"><a href="https://www.google.com/url?q=https://arxiv.org/abs/1903.00780&amp;sa=D&amp;ust=1589252852006000" class="c15">Fairness through pairwise comparison</a></span><span
    class="c0"> </span>

<!-- -->

-   <span class="c0">This paper recommends a pairwise fairness metric
    and also a strategy to improve fairness using pairwise
    regularization for recommender systems . This strategy is shown to
    have significantly improved fairness and we believe the same could
    be explored for our use case too . </span>

#### <span class="c23">Mitigation</span>

-   Constrained Optimization

<!-- -->

-   The TensorFlow Constrained Optimization (TFCO) library (github repo
    <span
    class="c1"><a href="https://www.google.com/url?q=https://github.com/google-research/tensorflow_constrained_optimization&amp;sa=D&amp;ust=1589252852007000" class="c15">here</a></span><span
    class="c0">) makes it easy to configure and train machine learning
    problems based on multiple different metrics (e.g. the precision on
    members of certain groups, the true positive rates on residents of
    certain countries etc).</span>
-   <span class="c0">Most of these metrics mentioned above are standard
    model evaluation metrics, however, TCFO offers the ability to
    minimize and constrain arbitrary combinations of them.</span>
-   We explored two metrics offered by the TCFO library - the Equalized
    Odds and Predictive Parity (as referenced & defined <span
    class="c1"><a href="https://www.google.com/url?q=https://ai.googleblog.com/2020/02/setting-fairness-goals-with-tensorflow.html&amp;sa=D&amp;ust=1589252852008000" class="c15">here</a></span><span
    class="c0">).</span>

<!-- -->

-   <span class="c14">Equalized Odds</span><span class="c0">: For any
    particular label and attribute, a classifier predicts that label
    equally well for all values of that attribute.</span>
-   <span class="c14">Predictive Parity</span><span class="c0">: A
    fairness metric that checks whether, for a given classifier, the
    precision rates are equivalent for subgroups under
    consideration.</span>

<!-- -->

-   <span class="c0">Even though we could not successfully use TCFO for
    mitigating the fairness concerns exposed by our base model, it did
    offer us an opportunity to assess a generalized optimizer
    library.</span>

<span class="c0"></span>

-   Debiasing word embeddings by adjust the directions of the word
    vectors as shared in <span
    class="c1"><a href="https://www.google.com/url?q=https://www.coursera.org/lecture/nlp-sequence-models/debiasing-word-embeddings-zHASj&amp;sa=D&amp;ust=1589252852008000" class="c15">https://www.coursera.org/lecture/nlp-sequence-models/debiasing-word-embeddings-zHASj</a></span>

<span class="c0"></span>

<span class="c13"></span>
=========================

<span class="c44">Challenges Faced</span>
=========================================

-   <span class="c0">Custom embeddings in feature columns: We wanted to
    take advantage of feature columns due to their close integration
    with many other TF tools, but were unable to as we wanted to use
    custom embedding not available on TF Hub.</span>
-   <span class="c0">Encoded CSV with TF fairness tools: We were unable
    to use TF fairness tools as the given examples use feature columns
    from raw CSV data. Instead we built our own using the ideas from
    Fairness Indicator APIs.</span>
-   <span class="c0">Non binary features fairness: Protected feature,
    Age for example had three values - young, middle age and old and
    current group fairness metrics like Equal Opportunity difference
    only work on 2 groups at a time. So we used our calculated best
    judgement based on false positives rates and compared various
    combinations two at a time.</span>
-   <span class="c0">Which fairness metric to use: Just like metrics for
    ML, we had to use our judgement based on the domain and use case to
    identify which fairness metrics are best suitable. We found equal
    opportunity score and average odds difference as relevant and easy
    to understand and hence used them.</span>
-   Many possibilities of bias: We had 15 ad categories, 2 protected
    features and 2 mitigation plans which is 60 possible combinations to
    evaluate.

<span class="c44">Suggestions to Tensorflow Tools</span><sup><a href="#cmnt1" id="cmnt_ref1">[a]</a></sup>
==========================================================================================================

<span class="c0">Nam liber tempor cum soluta nobis eleifend option
congue nihil imperdiet doming id quod mazim placerat facer possim assum.
Typi non habent claritatem insitam; est usus legentis in iis qui facit
eorum claritatem. Investigationes demonstraverunt lectores legere me
lius quod ii legunt saepius.</span>

<span class="c13"></span>
=========================

<span class="c7">Source Code</span>
===================================

-   Github Repository for our code: <span
    class="c1"><a href="https://www.google.com/url?q=https://github.com/salilkanitkar/responsible_ai_hackathon&amp;sa=D&amp;ust=1589252852010000" class="c15">https://github.com/salilkanitkar/responsible_ai_hackathon</a></span>
-   Training progress can be viewed on Tensorboard.dev at <span
    class="c1"><a href="https://www.google.com/url?q=https://tensorboard.dev/experiment/fkAOs09DSOKc52nhOtO1XA/&amp;sa=D&amp;ust=1589252852011000" class="c15">https://tensorboard.dev/experiment/fkAOs09DSOKc52nhOtO1XA/</a></span><span
    class="c0"> </span>
-   Fairness metrics dashboard <span
    class="c1"><a href="https://www.google.com/url?q=https://github.com/Nithanaroy/recommender-ai-fairness-dashboard&amp;sa=D&amp;ust=1589252852011000" class="c15">https://github.com/Nithanaroy/recommender-ai-fairness-dashboard</a></span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 592.50px; height: 443.00px;">![](images/image10.png)</span>

<span class="c7"></span>
========================

<span class="c7">References</span>
==================================

\[1\] <span
class="c1"><a href="https://www.google.com/url?q=https://ai.googleblog.com/2020/02/setting-fairness-goals-with-tensorflow.html&amp;sa=D&amp;ust=1589252852012000" class="c15">Setting Fairness Goals with the TensorFlow Constrained Optimization Library</a></span><span
class="c0"> </span>

\[2\] <span
class="c1"><a href="https://www.google.com/url?q=https://github.com/google-research/tensorflow_constrained_optimization&amp;sa=D&amp;ust=1589252852012000" class="c15">google-research/tensorflow_constrained_optimization</a></span><span
class="c0"> has code samples on constrained optimization</span>

\[3\] <span
class="c1"><a href="https://www.google.com/url?q=https://github.com/google-research/google-research/tree/master/pairwise_fairness&amp;sa=D&amp;ust=1589252852012000" class="c15">https://github.com/google-research/google-research/tree/master/pairwise_fairness</a></span><span
class="c0"> </span>

\[4\] <span
class="c1"><a href="https://www.google.com/url?q=https://github.com/tensorflow/fairness-indicators%23examples&amp;sa=D&amp;ust=1589252852013000" class="c15">https://github.com/tensorflow/fairness-indicators#examples</a></span>

<span class="c13"></span>

<span class="c0"></span>

<span class="c7">Submission Questions</span>
============================================

From <span
class="c1 c95"><a href="https://www.google.com/url?q=https://devpost.com/submit-to/9668-tf-2-2-challenge-building-ai-responsibly/start/submissions/new&amp;sa=D&amp;ust=1589252852013000" class="c15">https://devpost.com/submit-to/9668-tf-2-2-challenge-building-ai-responsibly/start/submissions/new</a></span>

<span class="c19 c14 c56">What's your project called?</span>

<span class="c0">Impact of user personality on advertisement
recommendations </span>

<span class="c0"></span>

<span class="c19 c14 c30">Here's the elevator pitch</span>

<span class="c45 c14 c56">What's your idea? This will be a short tagline
for the project. </span><span class="c14 c24">You can change this
later.</span>

<span class="c29">Marketing ads often misrepresent themselves and cater
to the audience in an unfair manner. We make sure our model exploring
the consumer's personality in recommending ads addresses this
concern.</span>

<span class="c30 c19 c14"></span>

<span class="c30 c19 c14">It’s built with</span>

<span class="c14 c56 c86">What languages, APIs, hardware, hosts,
libraries, UI Kits or frameworks are you using? </span><span
class="c14 c56 c65">You can change this later.</span>

<span class="c19 c14 c56">Language - </span><span
class="c18">Python</span>

<span class="c19 c14 c56">Editor</span><span class="c19 c62 c56"> -
Jupyter notebook</span>

<span class="c30 c19 c14">Tools / libraries </span>

-   <span class="c18">Development - Tensorflow, Keras, Pandas ,Numpy,
    sckit-learn</span>
-   <span class="c18">Visualization - seaborn, matplotlib, Tensorboard,
    </span>
-   <span class="c18">Fairness - Tensorflow, aif360, imblearn</span>
-   <span class="c18">Cloud services - Google Vision API</span>
-   <span class="c19 c62 c56">Dataset - </span><span
    class="c1"><a href="https://www.google.com/url?q=https://www.kaggle.com/groffo/ads16-dataset&amp;sa=D&amp;ust=1589252852015000" class="c15">https://www.kaggle.com/groffo/ads16-dataset</a></span> 

<span class="c0"></span>

<span class="c7">Submission Guidelines</span>
=============================================

<span class="c14">Due: May 11, 2020</span>

<span class="c2">Build a functioning Tensorflow 2.2 based solution, and
tell us about how you leveraged the Responsible AI practices as you did
so. </span>

-   <span class="c16">(</span><span class="c16 c87">Optional</span><span
    class="c16">) Submit a </span><span class="c14 c19">2-5 minute
    demo</span><span class="c2"> video hosted on YouTube, Vimeo, or
    Youku. Your video should include a demo of your working application,
    and any Responsible AI considerations and approaches.</span>
-   <span class="c16">Please submit at least </span><span
    class="c19 c14">one image or screenshot</span><span class="c2"> of
    your solution.</span>
-   <span class="c16">Please submit a </span><span class="c19 c14">PDF
    document </span><span class="c2">discussing the Responsible AI
    concerns, any tooling you used, any approaches you took to address
    these concerns, and any challenges you faced in the process. If you
    have any requests from Tensorflow tools, let us know that as well!
     </span>
-   <span class="c16">Make sure all of your code has been uploaded to a
    public repo on </span><span class="c19 c14">GitHub or another public
    repository</span><span class="c2">, and that a link to the repo has
    been included in your application.</span>

<span class="c16">All projects must be submitted by </span><span
class="c19 c14">May 11th, 2020</span><span class="c16">, at 11:45PM
Pacific Time. Judging will take place during Google I/O, from
</span><span class="c19 c14">May 11th</span><span class="c16"> through
</span><span class="c19 c14">May 15th</span><span class="c16">. Contest
winners will be informed of their project's status on the evening of
</span><span class="c19 c14">May 19th</span><span class="c2">.</span>

\_\_\_\_\_\_

<a href="#cmnt_ref1" id="cmnt1">[a]</a><span
class="c97 c69 c62 c91">+nithanaroy@hotmail.com</span>

<span class="c69 c62 c91 c97">\_Assigned to Nitin Pasumarthy\_</span>
