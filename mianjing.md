# Android 面试题

[toc]

## 1、网络

### 网络协议模型

应用层：负责处理特定的应用程序细节
HTTP、FTP、DNS

传输层：为两台主机提供端到端的基础通信
TCP、UDP

网络层：控制分组传输、路由选择等
IP

链路层：操作系统设备驱动程序、网卡相关接口

### TCP 和 UDP 区别

TCP 连接；可靠；有序；面向字节流；速度慢；较重量；全双工；适用于文件传输、浏览器等

全双工：A 给 B 发消息的同时，B 也能给 A 发
半双工：A 给 B 发消息的同时，B 不能给 A 发
UDP 无连接；不可靠；无序；面向报文；速度快；轻量；适用于即时通讯、视频通话等

TCP 三次握手
A：你能听到吗？
B：我能听到，你能听到吗？
A：我能听到，开始吧

A 和 B 两方都要能确保：我说的话，你能听到；你说的话，我能听到。所以需要三次握手

TCP 四次挥手
A：我说完了
B：我知道了，等一下，我可能还没说完
B：我也说完了
A：我知道了，结束吧

B 收到 A 结束的消息后 B 可能还没说完，没法立即回复结束标示，只能等说完后再告诉 A ：我说完了。

### POST 和 GET 区别

Get 参数放在 url 中；Post 参数放在 request Body 中
Get 可能不安全，因为参数放在 url 中

### HTTPS

HTTP 是超文本传输协议，明文传输；HTTPS 使用 SSL 协议对 HTTP 传输数据进行了加密

HTTP 默认 80 端口；HTTPS 默认 443 端口

优点：安全
缺点：费时、SSL 证书收费，加密能力还是有限的，但是比 HTTP 强多了

## 2、Java 基础&容器&同步&设计模式

### StringBuilder、StringBuffer、+、String.concat 链接字符串：

StringBuffer 线程安全，StringBuilder 线程不安全
+实际上是用 StringBuilder 来实现的，所以非循环体可以直接用 +，循环体不行，因为会频繁创建 StringBuilder
String.concat 实质是 new String ，效率也低，耗时排序：StringBuilder < StringBuffer < concat < +

### Java 泛型擦除

修饰成员变量等类结构相关的泛型不会被擦除
容器类泛型会被擦除

### ArrayList LinkedList

ArrayList

基于数组实现，查找快：o(1)，增删慢：o(n)
初始容量为10，扩容通过 System.arrayCopy 方法

LinkedList

基于双向链表实现，查找慢：o(n)，增删快：o(1)
封装了队列和栈的调用

### HashMap HashTable

HashMap

基于数组和链表实现，数组是 HashMap 的主体；链表是为解决哈希冲突而存在的
当发生哈希冲突且链表 size 大于阈值时会扩容，JAVA 8 会将链表转为红黑树提高性能
允许 key/value 为 null
HashTable

数据结构和 HashMap 一样
不允许 value 为 null
线程安全

### ArrayMap、SparseArray

ArrayMap

1.基于两个数组实现，一个存放 hash；一个存放键值对。扩容的时候只需要数组拷贝，不需要重建哈希表
2.内存利用率高
3.不适合存大量数据，因为会对 key 进行二分法查找（1000以下）

SparseArray

1.基于两个数组实现，int 做 key
2.内存利用率高
3.不适合存大量数据，因为会对 key 进行二分法查找（1000以下）

### volatile 关键字

只能用来修饰变量，适用修饰可能被多线程同时访问的变量
相当于轻量级的 synchronized，volatitle 能保证有序性（禁用指令重排序）、可见性；后者还能保证原子性
变量位于主内存中，每个线程还有自己的工作内存，变量在自己线程的工作内存中有份拷贝，线程直接操作的是这个拷贝
被 volatile 修饰的变量改变后会立即同步到主内存，保持变量的可见性。
双重检查单例，为什么要加 volatile？

1.volatile想要解决的问题是，在另一个线程中想要使用instance，发现instance!=null，但是实际上instance还未初始化完毕这个问题

2.将instance =newInstance();拆分为3句话是。1.分配内存2.初始化3.将instance指向分配的内存空

3.volatile可以禁止指令重排序，确保先执行2，后执行3

### wait 和 sleep

sleep 是 Thread 的静态方法，可以在任何地方调用
wait 是 Object 的成员方法，只能在 synchronized 代码块中调用，否则会报 IllegalMonitorStateException 非法监控状态异常
sleep 不会释放共享资源锁，wait 会释放共享资源锁

lock 和 synchronized

synchronized 是 Java 关键字，内置特性；Lock 是一个接口
synchronized 会自动释放锁；lock 需要手动释放，所以需要写到 try catch 块中并在 finally 中释放锁
synchronized 无法中断等待锁；lock 可以中断
Lock 可以提高多个线程进行读/写操作的效率
竞争资源激烈时，lock 的性能会明显的优于 synchronized

可重入锁
定义：已经获取到锁后，再次调用同步代码块/尝试获取锁时不必重新去申请锁，可以直接执行相关代码
ReentrantLock 和 synchronized 都是可重入锁

公平锁
定义：等待时间最久的线程会优先获得锁
非公平锁无法保证哪个线程获取到锁，synchronized 就是非公平锁
ReentrantLock 默认时非公平锁，可以设置为公平锁

乐观锁和悲观锁

悲观锁：线程一旦得到锁，其他线程就挂起等待，适用于写入操作频繁的场景；synchronized 就是悲观锁
乐观锁：假设没有冲突，不加锁，更新数据时判断该数据是否过期，过期的话则不进行数据更新，适用于读取操作频繁的场景
乐观锁 CAS：Compare And Swap，更新数据时先比较原值是否相等，不相等则表示数据过去，不进行数据更新
乐观锁实现：AtomicInteger、AtomicLong、AtomicBoolean

死锁 4 个必要条件
互斥
占有且等待
不可抢占
循环等待

synchronized 原理
每个对象都有一个监视器锁：monitor，同步代码块会执行 monitorenter 开始，motnitorexit 结束
wait/notify 就依赖 monitor 监视器，所以在非同步代码块中执行会报 IllegalMonitorStateException 异常

## 3、Java 虚拟机&内存结构&GC&类加载&四种引用&动态代理

JVM
定义：可以理解成一个虚构的计算机，解释自己的字节码指令集映射到本地 CPU 或 OS 的指令集，上层只需关注 Class 文件，与操作系统无关，实现跨平台
Kotlin 就是能解释成 Class 文件，所以可以跑在 JVM 上

JVM 内存模型
Java 多线程之间是通过共享内存来通信的，每个线程都有自己的本地内存
共享变量存放于主内存中，线程会拷贝一份共享变量到本地内存
volatile 关键字就是给内存模型服务的，用来保证内存可见性和顺序性

JVM 内存结构
线程私有：

1.程序计数器：记录正在执行的字节码指令地址，若正在执行 Native 方法则为空
2.虚拟机栈：执行方法时把方法所需数据存为一个栈帧入栈，执行完后出栈
3.本地方法栈：同虚拟机栈，但是针对的是 Native 方法

线程共享：

1.堆：存储 Java 实例，GC 主要区域，分代收集 GC 方法会吧堆划分为新生代、老年代
2.方法区：存储类信息，常量池，静态变量等数据

GC
回收区域：只针对堆、方法区；线程私有区域数据会随线程结束销毁，不用回收

回收类型：
1.堆中的对象

分代收集 GC 方法会吧堆划分为新生代、老年代
新生代：新建小对象会进入新生代；通过复制算法回收对象
老年代：新建大对象及老对象会进入老年代；通过标记-清除算法回收对象
2.方法区中的类信息、常量池

判断一个对象是否可被回收：
1.引用计数法
缺点：循环引用

2.可达性分析法
定义：从 GC ROOT 开始搜索，不可达的对象都是可以被回收的

GC ROOT
1.虚拟机栈/本地方法栈中引用的对象
2.方法区中常量/静态变量引用的对象

四种引用
强引用：不会被回收
软引用：内存不足时会被回收
弱引用：gc 时会被回收
虚引用：无法通过虚引用得到对象，可以监听对象的回收

ClassLoader
类的生命周期：

1.加载；2.验证；3.准备；4.解析；5.初始化；6.使用；7.卸载

类加载过程：

1.加载：获取类的二进制字节流；生成方法区的运行时存储结构；在内存中生成 Class 对象
2.验证：确保该 Class 字节流符合虚拟机要求
3.准备：初始化静态变量
4.解析：将常量池的符号引用替换为直接引用
5.初始化：执行静态块代码、类变量赋值

类加载时机：

1.实例化对象
2.调用类的静态方法
3.调用类的静态变量（放入常量池的常量除外）

类加载器：负责加载 class 文件

分类：

1.引导类加载器 - 没有父类加载器
2.拓展类加载器 - 继承自引导类加载器
3.系统类加载器 - 继承自拓展类加载器

双亲委托模型：
当要加载一个 class 时，会先逐层向上让父加载器先加载，加载失败才会自己加载

为什么叫双亲？不考虑自定义加载器，系统类加载器需要网上询问两层，所以叫双亲

判断是否是同一个类时，除了类信息，还必须时同一个类加载器

优点：

防止重复加载，父加载器加载过了就没必要加载了
安全，防止篡改核心库类
动态代理原理及实现
InvocationHandler 接口，动态代理类需要实现这个接口
Proxy.newProxyInstance，用于动态创建代理对象
Retrofit 应用： Retrofit 通过动态代理，为我们定义的请求接口都生成一个动态代理对象，实现请求

## 4、Android 基础&性能优化&Framwork

### Activity 启动模式

standard 标准模式
singleTop 栈顶复用模式，
推送点击消息界面
singleTask 栈内复用模式，
首页
singleInstance 单例模式，单独位于一个任务栈中
拨打电话界面
细节：
taskAffinity：任务相关性，用于指定任务栈名称，默认为应用包名
allowTaskReparenting：允许转移任务栈


### View 工作原理

DecorView (FrameLayout)
LinearLayout
titlebar
Content
调用 setContentView 设置的 View
ViewRoot 的 performTraversals 方法调用触发开始 View 的绘制，然后会依次调用:

performMeasure：遍历 View 的 measure 测量尺寸
performLayout：遍历 View 的 layout 确定位置
performDraw：遍历 View 的 draw 绘制

### 事件分发机制

一个 MotionEvent 产生后，按 Activity -> Window -> decorView -> View 顺序传递，View 传递过程就是事件分发，主要依赖三个方法:
dispatchTouchEvent：用于分发事件，只要接受到点击事件就会被调用，返回结果表示是否消耗了当前事件
onInterceptTouchEvent：用于判断是否拦截事件，当 ViewGroup 确定要拦截事件后，该事件序列都不会再触发调用此 ViewGroup 的 onIntercept
onTouchEvent：用于处理事件，返回结果表示是否处理了当前事件，未处理则传递给父容器处理
细节：
一个事件序列只能被一个 View 拦截且消耗
View 没有 onIntercept 方法，直接调用 onTouchEvent 处理
OnTouchListener 优先级比 OnTouchEvent 高，onClickListener 优先级最低
requestDisallowInterceptTouchEvent 可以屏蔽父容器 onIntercet 方法的调用
Window 、 WindowManager、WMS、SurfaceFlinger
Window：抽象概念不是实际存在的，而是以 View 的形式存在，通过 PhoneWindow 实现
WindowManager：外界访问 Window 的入口，内部与 WMS 交互是个 IPC 过程
WMS：管理窗口 Surface 的布局和次序，作为系统级服务单独运行在一个进程
SurfaceFlinger：将 WMS 维护的窗口按一定次序混合后显示到屏幕上
View 动画、帧动画及属性动画
View 动画：

作用对象是 View，可用 xml 定义，建议 xml 实现比较易读
支持四种效果：平移、缩放、旋转、透明度
帧动画：

通过 AnimationDrawable 实现，容易 OOM
属性动画：

可作用于任何对象，可用 xml 定义，Android 3 引入，建议代码实现比较灵活
包括 ObjectAnimator、ValuetAnimator、AnimatorSet
时间插值器：根据时间流逝的百分比计算当前属性改变的百分比
系统预置匀速、加速、减速等插值器
类型估值器：根据当前属性改变的百分比计算改变后的属性值
系统预置整型、浮点、色值等类型估值器
使用注意事项：
避免使用帧动画，容易OOM
界面销毁时停止动画，避免内存泄漏
开启硬件加速，提高动画流畅性 ，硬件加速：
将 cpu 一部分工作分担给 gpu ，使用 gpu 完成绘制工作
从工作分摊和绘制机制两个方面优化了绘制速度
Handler、MessageQueue、Looper
Handler：开发直接接触的类，内部持有 MessageQueue 和 Looper
MessageQueue：消息队列，内部通过单链表存储消息
Looper：内部持有 MessageQueue，循环查看是否有新消息，有就处理，没就阻塞
如何实现阻塞：通过 nativePollOnce 方法，基于 Linux epoll 事件管理机制
为什么主线程不会因为 Looper 阻塞：系统每 16ms 会发送一个刷新 UI 消息唤醒
MVC、MVP、MVVM
MVP：Model：处理数据；View：控制视图；Presenter：分离 Activity 和 Model
MVVM：Model：处理获取保存数据；View：控制视图；ViewModel：数据容器
使用 Jetpack 组件架构的 LiveData、ViewModel 便捷实现 MVVM
Serializable、Parcelable
Serializable ：Java 序列化方式，适用于存储和网络传输，serialVersionUID 用于确定反序列化和类版本是否一致，不一致时反序列化回失败
Parcelable ：Android 序列化方式，适用于组件通信数据传递，性能高，因为不像 Serializable 一样有大量反射操作，频繁 GC
Binder
Android 进程间通信的中流砥柱，基于客户端-服务端通信方式
使用 mmap 一次数据拷贝实现 IPC，传统 IPC：用户A空间->内核->用户B空间；mmap 将内核与用户B空间映射，实现直接从用户A空间->用户B空间
BinderPool 可避免创建多 Service
IPC 方式
Intent extras、Bundle：要求传递数据能被序列化，实现 Parcelable、Serializable ，适用于四大组件通信
文件共享：适用于交换简单的数据实时性不高的场景
AIDL：AIDL 接口实质上是系统提供给我们可以方便实现 BInder 的工具
Android Interface Definition Language，可实现跨进程调用方法
服务端：将暴漏给客户端的接口声明在 AIDL 文件中，创建 Service 实现 AIDL 接口并监听客户端连接请求
客户端：绑定服务端 Service ，绑定成功后拿到服务端 Binder 对象转为 AIDL 接口调用
RemoteCallbackList 实现跨进程接口监听，同个 Binder 对象做 key 存储客户端注册的 listener
监听 Binder 断开：1.Binder.linkToDeath 设置死亡代理；2. onServiceDisconnected 回调
Messenger：基于 AIDL 实现，服务端串行处理，主要用于传递消息，适用于低并发一对多通信
ContentProvider：基于 Binder 实现，适用于一对多进程间数据共享
Socket：TCP、UDP，适用于网络数据交换
Android 系统启动流程
按电源键 -> 加载引导程序 BootLoader 到 RAM -> 执行 BootLoader 程序启动内核 -> 启动 init 进程 -> 启动 Zygote 和各种守护进程 ->
启动 System Server 服务进程开启 AMS、WMS 等 -> 启动 Launcher 应用进程
App 启动流程
Launcher 中点击一个应用图标 -> 通过 AMS 查找应用进程，若不存在就通过 Zygote 进程 fork

进程保活
进程优先级：1.前台进程 ；2.可见进程；3.服务进程；4.后台进程；5.空进程
进程被 kill 场景：1.切到后台内存不足时被杀；2.切到后台厂商省电机制杀死；3.用户主动清理
保活方式：
1.Activity 提权：挂一个 1像素 Activity 将进程优先级提高到前台进程
2.Service 提权：启动一个前台服务（API>18会有正在运行通知栏）
3.广播拉活
4.Service 拉活
5.JobScheduler 定时任务拉活
6.双进程拉活
网络优化及检测
速度：1.GZIP 压缩（okhttp 自动支持）；2.Protocol Buffer 替代 json；3.优化图片/文件流量；4.IP 直连省去 DNS 解析时间
成功率：1.失败重试策略；
流量：1.GZIP 压缩（okhttp 自动支持）；2.Protocol Buffer 替代 json；3.优化图片/文件流量；5.文件下载断点续传 ；6.缓存
协议层的优化，比如更优的 http 版本等
监控：Charles 抓包、Network Monitor 监控流量
UI卡顿优化
减少布局层级及控件复杂度，避免过度绘制
使用 include、merge、viewstub
优化绘制过程，避免在 Draw 中频繁创建对象、做耗时操作
内存泄漏场景及规避
1.静态变量、单例强引跟生命周期相关的数据或资源，包括 EventBus
2.游标、IO 流等资源忘记主动释放
3.界面相关动画在界面销毁时及时暂停
4.内部类持有外部类引用导致的内存泄漏

handler 内部类内存泄漏规避：1.使用静态内部类+弱引用 2.界面销毁时清空消息队列
检测：Android Studio Profiler
LeakCanary 原理
通过弱引用和引用队列监控对象是否被回收
比如 Activity 销毁时开始监控此对象，检测到未被回收则主动 gc ，然后继续监控
OOM 场景及规避
加载大图：减小图片
内存泄漏：规避内存泄漏

## 5、Android 模块化&热修复&热更新&打包&混淆&压缩

Dalvik 和 ART
Dalvik
谷歌设计专用于 Android 平台的 Java 虚拟机，可直接运行 .dex 文件，适合内存和处理速度有限的系统
JVM 指令集是基于栈的；Dalvik 指令集是基于寄存器的，代码执行效率更优
ART
Dalvik 每次运行都要将字节码转换成机器码；ART 在应用安装时就会转换成机器码，执行速度更快
ART 存储机器码占用空间更大，空间换时间
APK 打包流程
1.aapt 打包资源文件生成 R.java 文件；aidl 生成 java 文件
2.将 java 文件编译为 class 文件
3.将工程及第三方的 class 文件转换成 dex 文件
4.将 dex 文件、so、编译过的资源、原始资源等打包成 apk 文件
5.签名
6.资源文件对齐，减少运行时内存

App 安装过程
首先要解压 APK，资源、so等放到应用目录
Dalvik 会将 dex 处理成 ODEX ；ART 会将 dex 处理成 OAT；
OAT 包含 dex 和安装时编译的机器码
组件化路由实现
ARoute：通过 APT 解析 @Route 等注解，结合 JavaPoet 生成路由表，即路由与 Activity 的映射关系

## 6、音视频&FFmpeg&播放器

FFmpeg
基于命令方式实现了一个音视频编辑 App：
https://github.com/yhaolpz/FFmpegCmd

集成编译了 AAC、MP3、H264 编码器

播放器原理
视频播放原理：（mp4、flv）-> 解封装 -> （mp3/aac、h264/h265）-> 解码 -> （pcm、yuv）-> 音视频同步 -> 渲染播放

音视频同步：

选择参考时钟源：音频时间戳、视频时间戳和外部时间三者选择一个作为参考时钟源（一般选择音频，因为人对音频更敏感，ijk 默认也是音频）
通过等待或丢帧将视频流与参考时钟源对齐，实现同步
IjkPlayer 原理
集成了 MediaPlayer、ExoPlayer 和 IjkPlayer 三种实现，其中 IjkPlayer 基于 FFmpeg 的 ffplay

音频输出方式：AudioTrack、OpenSL ES；视频输出方式：NativeWindow、OpenGL ES


--- 

# kotlin 刷题


for loop

```kotlin
for (item in collection) print(item)

for (i in 1..3) {
    println(i)
}
for (i in 6 downTo 0 step 2) {
    println(i)
}
for (i in array.indices) {
    println(array[i])
}
for ((index, value) in array.withIndex()) {
    println("the element at $index is $value")
}
``` 

IntArray

```
val nums = IntArray(10)
nums[1] = 2
nums.size
```

