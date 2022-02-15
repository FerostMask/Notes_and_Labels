# 第二章

学习链接：

[@Override](https://blog.csdn.net/upc1607020107/article/details/81274398)

[重写父类方法 - 懒虫小园 - 博客园 (cnblogs.com)](https://www.cnblogs.com/hwtfamily/p/9092988.html)

[读懂Python中的self](https://blog.csdn.net/xrinosvip/article/details/89647884)

### required关键字

命名参数中必须要传的参数可以通过required关键字指定

### State类

>   关于State<T extends StatefulWiget> with Diagnosticable还是不太理解，英文也不太好阅读，拿出来试着翻译下

[State] objects are created by the framework by calling the [StatefulWidget.createState] method when inflating a [StatefulWidget] to insert it into the tree

当你为了填充[StatefulWidget]而把它插入到树中的同时，框架会通过调用[StatefulWidget.createState]来创建[State]对象（ 翻完还是看不懂<-是不是翻错了

### State中的setState方法

调用setState方法会通知Flutter框架当前State中有些东西发生了改变，这会重跑一遍当前State中的build方法，于是我们就能看到更新后的内容反应在界面中。如果State中某些东西发生了改变而我们又没有调用setState方法，那么很可能什么都不会发生。

