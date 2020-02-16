# sort

[toc]


![](https://pic.leetcode-cn.com/cde64bf682850738153e6c76dd3f6fb32201ce3c73c23415451da1eead9eb7cb-20190624173156.jpg)


链接：https://leetcode-cn.com/problems/top-k-frequent-elements/solution/leetcode-di-347-hao-wen-ti-qian-k-ge-gao-pin-yuan-/

## 912. Sort an Array

https://leetcode-cn.com/problems/sort-an-array/

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
```

### Quick Sort

快速排序 https://blog.csdn.net/willshine19/article/details/52565739

```python
def quick_sort(nums):
    length = len(nums)
    if length <= 1:
        return nums
    pivot = nums.pop()
    greater, lesser = [], []
    for num in nums:
        if num > pivot:
            greater.append(num)
        else:
            lesser.append(num)
    return quick_sort(lesser) + [pivot] + quick_sort(greater)

```

```java
private static void quickSort(int[] array, int start, int end) {
    if (start < end) {
        int mid = partition(array, start, end);
        quickSort(array, start, mid - 1);
        quickSort(array, mid + 1, end);
    }
}

private static int partition(int[] src, int start, int end) {
    int target = src[start];
    while (start < end) {
        while (start < end && src[end] >= target) {
            end--;
        }
        src[start] = src[end];
        while (start < end && src[start] <= target) {
            start++;
        }
        src[end] = src[start];
    }
    // 此时 start == end
    src[start] = target;
    return start;
}
```

### Merge Sort

归并排序 https://blog.csdn.net/willshine19/article/details/52663843

```python
def merge_sort(nums):
    def merge(left, right):
        result = []
        while left and right:
            result.append((left if left[0] <= right[0] else right).pop(0))
        return result + left + right
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    return merge(merge_sort(nums[:mid]), merge_sort(nums[mid:]))
```

### Bubble Sort

```python
def bubble_sort(nums):
    length = len(nums)
    for i in range(length-1):
        swapped = False
        for j in range(length-1-i):
            if nums[j] > nums[j+1]:
                swapped = True
                nums[j], nums[j+1] = nums[j+1], nums[j]
        if not swapped: break  # Stop iteration if the nums is sorted.
    return nums
```

### Insertion Sort

```python
def insertion_sort(nums):
    for i in range(1, len(nums)):
        insertion_index = i
        while insertion_index > 0 and nums[insertion_index - 1] > nums[insertion_index]:
            nums[insertion_index], nums[insertion_index - 1] = nums[insertion_index - 1], nums[insertion_index]
            insertion_index -= 1

    return nums
```

### Heap Sort

堆排序 处理海量数据的 `topK`，`分位数` 非常合适，
因为它不用将所有的元素都进行排序，只需要比较和根节点的大小关系就可以了，同时也不需要一次性将所有的数据都加载到内存。


[黄浩杰heapsort](https://www.bilibili.com/video/av47196993?from=search&seid=773170647041705462)


```
 (i-1)/2
    |
    i
  /   \
2i+1  2i+2
```

数据结构 heap:
1. complete binary tree
1. parent > children

heapify: 对一个节点做 heapify, 确保这节点 > children, 如果发生交换, 还要向下继续 heapify

build heap: 对一个树的所有父节点做 heapify (从最后到最前), 根节点就是最大值(最大堆)

最大堆 大根堆: 根是最大值
最小堆 小根堆: 根是最小值 

heap sort:

https://www.geeksforgeeks.org/heap-sort/

nlogn

```python

# Python program for implementation of heap Sort 
  
# To heapify subtree rooted at index i. 
# n is size of heap 
def heapify(arr, n, i): 
    max_i = i # Initialize largest as root 
    l = 2 * i + 1     # left = 2*i + 1 
    r = 2 * i + 2     # right = 2*i + 2 
  
    # See if left child of root exists and is 
    # greater than root 
    if l < n and arr[i] < arr[l]: 
        max_i = l 
  
    # See if right child of root exists and is 
    # greater than root 
    if r < n and arr[max_i] < arr[r]: 
        max_i = r 
  
    # Change root, if needed 
    if max_i != i: 
        arr[i], arr[max_i] = arr[max_i],a rr[i] # swap 
  
        # Heapify the root. 
        heapify(arr, n, max_i) 
  
# The main function to sort an array of given size 
def heapSort(arr): 
    n = len(arr) 
  
    # Build a maxheap. 
    for i in range(n, -1, -1): 
        heapify(arr, n, i) 
  
    # One by one find max elements 
    for i in range(n-1, 0, -1): 
        arr[i], arr[0] = arr[0], arr[i] # swap 
        heapify(arr, i, 0) 
  
# Driver code to test above 
arr = [ 12, 11, 13, 5, 6, 7] 
heapSort(arr) 
n = len(arr) 
print ("Sorted array is") 
for i in range(n): 
    print ("%d" %arr[i]), 
# This code is contributed by Mohit Kumra 
```

703. 数据流中的第K大元素
https://leetcode-cn.com/problems/kth-largest-element-in-a-stream

```python
'''
heap
'''
class KthLargest:
    def __init__(self, k: int, nums):
        self.k = k
        if len(nums) > k:
            # 最大的 k 的数 
            self.h = heapq.nlargest(k, nums)
        else:
            self.h = nums[:]
        heapq.heapify(self.h)
        # 找出最小值 放在头部

    def add(self, val: int) -> int:
        heapq.heappush(self.h, val)
        if len(self.h) > self.k:
            heapq.heappop(self.h)
        return self.h[0] # k个数字中的最小值
```

347. 前 K 个高频元素

https://leetcode-cn.com/problems/top-k-frequent-elements/

时间复杂度：O(nlogk)，n 表示数组的长度。首先，遍历一遍数组统计元素的频率，这一系列操作的时间复杂度是 O(n)；接着，遍历用于存储元素频率的 map，如果元素的频率大于最小堆中顶部的元素，则将顶部的元素删除并将该元素加入堆中，这里维护堆的数目是 kk，所以这一系列操作的时间复杂度是 O(nlogk) 的；因此，总的时间复杂度是 O(nlog⁡k)。


```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        def heapify(arr, n, i):
            smallest = i  # 构造根节点与左右子节点
            l = 2 * i + 1
            r = 2 * i + 2
            if l < n and arr[l][1] < arr[i][1]:  # 如果左子节点在范围内且小于父节点
                smallest = l
            if r < n and arr[r][1] < arr[smallest][1]:
                smallest = r
            if smallest != i:  # 递归基:如果没有交换，退出递归
                arr[i], arr[smallest] = arr[smallest], arr[i]
                heapify(arr, n, smallest)  # 确保交换后，小于其左右子节点

        # 哈希字典统计出现频率
        map_dict = {}
        for item in nums:
            if item not in map_dict.keys():
                map_dict[item] = 1
            else:
                map_dict[item] += 1

        map_arr = list(map_dict.items())
        lenth = len(map_dict.keys())
        # 构造规模为k的minheap
        if k <= lenth:
            k_minheap = map_arr[:k]
            # 从后往前建堆，避免局部符合而影响递归跳转，例:2,1,3,4,5,0
            for i in range(k // 2 - 1, -1, -1): 
                heapify(k_minheap, k, i)
            # 对于k:, 大于堆顶则入堆，维护规模为k的minheap
            for i in range(k, lenth): # 堆建好了，没有乱序，从前往后即可
                if map_arr[i][1] > k_minheap[0][1]:
                    k_minheap[0] = map_arr[i] # 入堆顶
                    heapify(k_minheap, k, 0)  # 维护 minheap
        # 如需按顺序输出，对规模为k的堆进行排序
        # 从尾部起，依次与顶点交换再构造minheap，最小值被置于尾部
        for i in range(k - 1, 0, -1):
            k_minheap[i], k_minheap[0] = k_minheap[0], k_minheap[i]
            k -= 1 # 交换后，维护的堆规模-1
            heapify(k_minheap, k, 0)
        return [item[0] for item in k_minheap]
```

---

## 148. Sort List

https://www.lintcode.com/problem/sort-list/description

https://leetcode-cn.com/problems/sort-list/

Sort a linked list in O(n log n) time using constant space complexity.

冒泡排序 超时

```python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # 2019.7.8
        dummy = ListNode(0)
        while head:
            this = dummy
            while this.next:
                if head.val <= this.next.val:
                    head_next = head.next
                    head.next = this.next
                    this.next = head
                    head = head_next
                    break
                else:
                    this = this.next
            else:
                this.next = head
                head = head.next
                this.next.next = None
        return dummy.next 
```

```java
public class Solution {
    public ListNode sortList(ListNode head) {  
        // 2015-5-24 O(nlogn) 分治法 递归 类似归并排序
        // exit condition
        if (head == null || head.next == null) {
            return head;
        }
        
        ListNode mid = findMid(head);
        ListNode listB = sortList(mid.next);
        mid.next = null;
        ListNode listA = sortList(head);
        // listB 不长于 listA
        
        return mergeList(listA, listB);
        
    }
    
    private ListNode findMid(ListNode head) {
        if (head == null) {
            return head;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
    
    private ListNode mergeList(ListNode listA, ListNode listB) {
        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;
        while (listA != null && listB != null) {
            if (listA.val < listB.val) {
                tail.next = listA;
                listA = listA.next;
            } else {
                tail.next = listB;
                listB = listB.next;
            }
            tail = tail.next;
        }
        if (listA != null) {
            tail.next = listA;
        }
        if (listB != null) {
            tail.next = listB;
        }
        return dummy.next;
    }
}
```

### Insertion Sort List

https://leetcode-cn.com/problems/insertion-sort-list/

```java
public class Solution {
    public ListNode insertionSortList(ListNode head) {
        // 2015-08-28 O(n^2)
        // 建立新的dummy链
        ListNode dummy = new ListNode(0);
        
        // 遍历两个链表，一定是两个while循环嵌套        
        // 遍历未排序的链
        while (head != null) {
            // 从头遍历dummy链，找合适的插入位置
            ListNode insertPos = dummy;
            while (insertPos.next != null && insertPos.next.val < head.val) {
                insertPos = insertPos.next;
            }
            
            // 找到插入位置，在insertPoc.next的位置插入head节点
            ListNode temp = head.next;
            head.next = insertPos.next;
            insertPos.next = head;
            head = temp;
        }
 
        return dummy.next;
    }
}
```

## 283. Move Zeroes

https://leetcode-cn.com/problems/move-zeroes/

Version 1

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        # left 第一个 0
        # right 是 left 后面第一个非0
        left = right = 0
        ln = len(nums)
        
        for left, val in enumerate(nums):
            if val != 0:
                continue
            # 找到了 left

            if left >= right:
                right = left + 1
            
            while right < ln and nums[right] == 0: 
                right += 1
            if right == ln: return
            # 找到了 right
            
            nums[left] = nums[right]
            nums[right] = 0
            right += 1
```

Version 2

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        pos = 0
        for num in nums:
            if num != 0: nums[pos] = num; pos += 1
        while pos < len(nums):
            nums[pos] = 0; pos += 1
```

## 75. Sort Colors

medium https://leetcode-cn.com/problems/sort-colors/

Version 1 O(n) 遍历两次

```java
class Solution {
    public void sortColors(int[] nums) {
        // 2015-09-15
        if (nums == null || nums.length == 0) {
            return;
        }
        
        int count0 = 0;
        int count1 = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                count0++;
            } else if (nums[i] == 1) {
                count1++;
            }
        }
        
        for (int i = 0; i < nums.length; i++) {
            if (count0 > 0) {
                nums[i] = 0;
                count0--;
            } else if (count1 > 0) {
                nums[i] = 1;
                count1--;
            } else {
                nums[i] = 2;
            }
        }
        
        return;
    }
}
```

Version 2 遍历1次

```java
class Solution {
    public void sortColors(int[] nums) {
        // 2015-09-15 O(n)
        if(nums == null || nums.length <= 1)
            return;
        
        // pl 指向0后一个数
        int pl = 0;
        // pr 指向2前一个数
        int pr = nums.length - 1;
        int i = 0;
        while(i <= pr){
            if(nums[i] == 0){
                // 遇到0 换到前面
                swap(nums, pl, i);
                pl++;
                i++;
            }else if(nums[i] == 1){
                i++;
            }else{
                // 遇到2 换到后面
                swap(nums, pr, i);
                pr--;
            }
        }
    }
    
    private void swap(int[] a, int i, int j) {
        if (i == j) {
            return;
        }
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}
```

## Partition Array

https://www.lintcode.com/problem/partition-array/description

Given an array nums of integers and an int k, partition the array (i.e move the elements in "nums") such that:

All elements < k are moved to the left
All elements >= k are moved to the right

```java
public class Solution {
    public int partitionArray(int[] nums, int k) {
	    // 2015-09-23 O(n) 
	    // 注意所有元素小于 k 的情况
	    if (nums == null || nums.length < 1) {
	        return 0;
	    }
	    
	    int start = 0;
	    int end = nums.length - 1;
	    while (start < end) {
	        if (nums[start] < k) {
	            start++;
	        } else if (nums[end] >= k) {
	            end--;
	        } else {
	            swap(nums, start, end);
	            start++;
	        }
	    }
	    // 此时 start = end
	    if (nums[start] < k) {
	        return start + 1;
	    } else {
	        return start;
	    }
    }
    
    private void swap(int[] nums, int start, int end) {
        int temp = nums[start];
        nums[start] = nums[end];
        nums[end] = temp;
    }
}
```
