# Domain Adaptation

本质是一个二元分类器

![](.Adaptation_images/32fd4b27.png)

Domain Adaptation技术,也可以看做是 Transfer Learning 的一种
在A任务上学习的技能可以用在B上，一个Domain上学到的用在另一个Domain上

## Domain Shift

![](.Adaptation_images/699ecccb.png)

![](.Adaptation_images/e8e2cb28.png)

只有少许标注需要做Adaptation

![](.Adaptation_images/89df5e67.png)

---

**怎么用没有标注的资料在Source Domain上训练并用在Target Domain上？**

![](.Adaptation_images/b08c8b85.png)
**把不一样的地方去掉，只抽取一样的部分。比如去掉颜色，Feature Extractor (network)，最后生成的feature是一样的**


---

**怎么找出这样的一个Feature Extractor呢？**

![](.Adaptation_images/52161943.png)
**把一个分类器分成Feature Extractor和Label Predictor两部分**

![](.Adaptation_images/2d02a456.png)
