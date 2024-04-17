# Array

[TOC]

## 1. Two Sum

Easy https://leetcode-cn.com/problems/two-sum/

Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Version 1 暴力 O(n^2)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 2019.7.28
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
        else:
            return []
```

Version 2 快排 + 二分查找  时间 o(nlogn)

Version 3 哈希表 时间 O(n) 空间 O(n)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 2019.7.28
        d = {}
        for i, num in enumerate(nums):
            if target - num in d:
                return [d[target - num], i]
            d[num] = i
        else:
            return []
```

```java
public class Solution {
    public int[] twoSum(int[] numbers, int target) {
        // 2015-09-25 O(n) 时间
        if (numbers == null || numbers.length == 0) {
            return new int[2];
        }
        int[] rst = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        int count = 1;
        for (int i = 0; i < numbers.length; i++) {
            map.put(numbers[i], count++);
        }
        for (Integer temp : map.keySet()) {
            if (map.containsKey(target - temp) 
                && map.get(temp) != map.get(target - temp)) {
                rst[0] = Math.min(map.get(temp), map.get(target - temp));
                rst[1] = Math.max(map.get(temp), map.get(target - temp));
                return rst;
            }
        }
        return new int[2];
    }
}
```

## 15.3 Sum

Medium https://leetcode-cn.com/problems/3sum/

Given an array nums of n integers, are there elements a, b, c in nums such that `a + b + c = 0`? 

Find **all** unique triplets in the array which gives the sum of zero.

方法1

DFS

```java
public class Solution {
    public ArrayList<ArrayList<Integer>> threeSum(int[] numbers) {
        // 2015-10-14 DFS 
        ArrayList<ArrayList<Integer>> rst = new ArrayList<>();
        if (numbers == null || numbers.length == 0) {
            return rst;
        }
        Arrays.sort(numbers);
        ArrayList<Integer> list = new ArrayList<>();
        helper(numbers, rst, list, 0);
        return rst;
    }
    
    private void helper(int[] numbers, ArrayList<ArrayList<Integer>> rst, 
            ArrayList<Integer> list, int pos) {
        if (list.size() == 3) {
            if (list.get(0) + list.get(1) + list.get(2) == 0 
                    && !rst.contains(list)) {
                rst.add(new ArrayList<Integer>(list));
            }
            return;
        }
        for (int i = pos; i < numbers.length; i++) {
            list.add(numbers[i]);
            helper(numbers, rst, list, i + 1);
            list.remove(list.size() - 1);
        }
    }
}
```

方法2



思路: 
1. 排序
1. 遍历选择第一个数
1. 从第一个数右边的子数组中选择第二三个数
1. left 最左 right 最右, 若sum<0 则 left++, 如果sum >0 则right--

## 16. 最接近的三数之和

https://leetcode-cn.com/problems/3sum-closest/

## 136. Single Number

Easy https://leetcode-cn.com/problems/single-number/

![](https://pic.leetcode-cn.com/Figures/137/methods.png)

Given a non-empty array of integers, every element appears **twice** except for one. Find that single one.

```python
'''
map
时间 O(n)
'''
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        lookup = set()
        for i in nums:
            if i in lookup:
                lookup.remove(i)
            else:
                lookup.add(i)
        return lookup.pop()
```

```python
'''
求和
时间 O(n)
'''
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        unisum = sum(set(nums))
        return unisum * 2 - sum(nums)
```


```python
'''
时间 O(n)
空间 O(1)
'''
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        xor = 0
        for n in nums:
            xor ^= n
        return xor
```

## 137. Single Number II

Medium https://leetcode-cn.com/problems/single-number-ii/

Given a non-empty array of integers, every element appears **three times** except for one, which appears exactly once. Find that single one.

方法一

```python
'''
map
时间 O(n)
'''
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        from collections import Counter
        lookup = Counter()
        for i in nums:
            lookup[i] += 1
        for k, v in lookup.items():
            if v == 1:
                return k
```

方法二

```python
'''
求和
时间 O(n)
'''
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        set_nums = set(nums)
        return (3 * sum(set_nums) - sum(nums)) // 2
```

```java
public class Solution {
	/**
	 * @param A : An integer array
	 * @return : An integer 
	 */
    public int singleNumberII(int[] A) {
        // 2015-09-17 O(n)
        if (A == null || A.length == 0) {
            return 0;
        }
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < A.length; i++) {
            if (!map.containsKey(A[i])) {
                map.put(A[i], 1);
            } else if (map.get(A[i]) == 2) {
                map.remove(A[i]);
            } else {
                map.put(A[i], map.get(A[i]) + 1);
            }
        }
        Set<Integer> keyset = map.keySet();
        for (Integer rst : keyset) {
            return rst;
        }
        return 0;
    }
}
```

方法三

```python
'''
模拟三进制
时间 O(n)
空间 O(1)
'''
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        b1, b2 = 0, 0 # 出现一次的位，和两次的位
        for n in nums:
            b1 = (b1 ^ n) & ~ b2 # 既不在出现一次的b1，也不在出现两次的b2里面，我们就记录下来，出现了一次，再次出现则会抵消
            b2 = (b2 ^ n) & ~ b1 # 既不在出现两次的b2里面，也不再出现一次的b1里面(不止一次了)，记录出现两次，第三次则会抵消
        return b1
```


## 260. Single Number III

Medium https://leetcode-cn.com/problems/single-number-iii/

Given an array of numbers `nums`, in which exactly two elements appear only **once** and all the other elements appear exactly **twice**. Find the two elements that appear only once.

方法一 

```python
'''
map
时间 O(n)
'''
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        from collections import Counter
        lookup = Counter()
        for i in nums:
            lookup[i] += 1
        ans = []
        for k, v in lookup.items():
            if v == 1:
                ans.append(k)
        return ans
```


方法二

```python
'''
分成两组
时间 O(N) 空间 O(1)
'''
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        xor = 0;
        for num in nums:
            xor ^= num
        xor = xor & ~(xor - 1)
        arr1, arr2 = [], []
        for num in nums:
            if xor & num == 0:
                arr1.append(num)
            else:
                arr2.append(num)
        num1, num2 = 0, 0
        for num in arr1:
            num1 ^= num
        for num in arr2:
            num2 ^= num
        return [num1, num2]
```

## 169. 多数元素

169. Majority Element

Easy https://leetcode-cn.com/problems/majority-element/

Given an array of size `n`, find the majority element. The majority element is the element that appears more than `⌊ n/2 ⌋` times.

Version 1

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        from collections import Counter
        d = Counter()
        for i, v in enumerate(nums):
            d[v] += 1
        n = len(nums)
        for k, v in d.items():
            if v > n / 2: return k
        return None
```

Version 2 时间 O(n) 空间 O(1)

```java
public class Solution {
    public int majorityNumber(ArrayList<Integer> nums) {
        // 2015-09-07 不同的数相互抵消，剩下majority number
        int count = 0;
        int candidate = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (count == 0) {
                candidate = nums.get(i);
                count++;
                continue;
            }
            if (nums.get(i) == candidate) {
                count++;
            } else {
                count--;
            }
        }
        return candidate;
    }
}
```


## 229. 求众数 II

229. Majority Element II

Medium https://leetcode-cn.com/problems/majority-element-ii/

Given an integer array of size n, find all elements that appear more than `⌊ n/3 ⌋` times.


```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        from collections import Counter
        d = Counter()
        for i, v in enumerate(nums):
            d[v] += 1
        n = len(nums)
        res = []
        for k, v in d.items():
            if v > n / 3: res.append(k)
        return res
```


```java
public class Solution {
    public int majorityNumber(ArrayList<Integer> nums) {
        // 2015-09-07
        if (nums == null || nums.size() == 0) {
            return -1;
        }
        int candidate1 = 0;
        int candidate2 = 0;
        int count1 = 0;
        int count2 = 0;
        for (int i = 0; i < nums.size(); i++) {
            // 注意 if 的顺序，确保 candidate1 和 candidate2 不是同一个数
            if (count1 == 0) {
                candidate1 = nums.get(i);
                count1++;
            } else if (candidate1 == nums.get(i)) {
                count1++;
            } else if (count2 == 0) {
                candidate2 = nums.get(i);
                count2++;
            } else if (candidate2 == nums.get(i)) {
                count2++;
            } else {
                count1--;
                count2--;
            }
        }
        
        // 此时只剩下两个数，candidate1和candidate2
        count1 = 0;
        count2 = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (candidate1 == nums.get(i)) {
                count1++;
            }
            if (candidate2 == nums.get(i)) {
                count2++;
            }
        }
        return count1 > count2 ? candidate1 : candidate2;
    }
}
```

## Majority Number III

Medium https://www.lintcode.com/problem/majority-number-iii/description

Given an array of integers and a number k, the majority number is the number that occurs more than 1/k of the size of the array.

https://blog.csdn.net/willshine19/article/details/48649743

## 53. 最大子序和

53. Maximum Subarray

Easy https://leetcode-cn.com/problems/maximum-subarray/

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Version 1 时间 O(n^2) 暴力

Version 2 时间 O(n) 空间 O(1)

```java
public class Solution {
    public int maxSubArray(ArrayList<Integer> nums) {
        // 2015-09-09 O(n)
        if (nums == null || nums.size() == 0) {
            // 异常
        }
        int sum = 0;
        int minSum = 0;
        int rst = Integer.MIN_VALUE;
        
        for (int i = 0; i < nums.size(); i++) {
            sum += nums.get(i);
            rst = Math.max(rst, sum - minSum);
            minSum = Math.min(minSum, sum);
        }
        
        return rst;
    }
}
```

Version 3

```python
'''
动态规划
dp[n], dp[i] 表示以 nums[i] 为结尾的子序的最大和 或 0
时间 O(n)
'''
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums: return 0
        n = len(nums)
        dp = [0] * n
        dp[0] = max(nums[0], 0)
        for i in range(1, n):
            dp[i] = max(dp[i - 1] + nums[i], 0)    
        return max(dp) if max(dp) > 0 else max(nums)
```



## 172. Factorial Trailing Zeroes

https://leetcode-cn.com/problems/factorial-trailing-zeroes/

Given an integer `n`, return the number of trailing zeroes in `n!`.

```java
class Solution {
    public long trailingZeros(long n) {
        // 2015-09-09
        // n的阶乘的因子中有几个5
        // 因为2肯定比5多，所以不用考虑2
        long rst = 0;
        while ((n / 5) !=0) {
            n /= 5;
            rst += n;
        }
        return rst;
    }
};
```

## Subarray Sum

https://www.lintcode.com/problem/subarray-sum/description

Given an integer array, find a subarray where the sum of numbers is zero.

Version 1 

```java
public class Solution {
    public ArrayList<Integer> subarraySum(int[] nums) {
        // 2015-09-06 暴力 O(n^2)
        ArrayList<Integer> rst = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            int sum = 0;
            for (int j = i; j < nums.length; j++) {
                sum += nums[j];
                if (sum == 0) {
                    rst.add(i);
                    rst.add(j);
                    return rst;
                }
            }
        }
        return rst;
    }
}
```

Verison 2 用哈希表降低时间复杂度

```java
public class Solution {
    public ArrayList<Integer> subarraySum(int[] nums) {
        // write your code here
        int len = nums.length;
       
        ArrayList<Integer> ans = new ArrayList<Integer>();
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
       
        map.put(0, -1);
       
        int sum = 0;
        for (int i = 0; i < len; i++) {
            sum += nums[i];
           
            if (map.containsKey(sum)) {
                ans.add(map.get(sum) + 1);
                ans.add(i);
                return ans;
            }
            
            map.put(sum, i);
        }
       
        return ans;
    }
}
```

## 1089. Duplicate Zeros

https://leetcode-cn.com/problems/duplicate-zeros/

Input: [1,0,2,3,0,4,5,0]
Output: null
Explanation: After calling your function, the input array is modified to: [1,0,0,2,3,0,0,4]

```python
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        zero_count = 0
        for val in arr:
            if val == 0:
                zero_count += 1
        
        if zero_count == 0:
            return
        
        for idx in reversed(range(len(arr))):
            if arr[idx] == 0:
                zero_count -= 1
                if idx + zero_count + 1 < len(arr):
                    arr[idx + zero_count + 1] = 0
            if idx + zero_count < len(arr):
                arr[idx + zero_count] = arr[idx]
                
```

## Subarray Sum Closest

Medium https://www.lintcode.com/problem/subarray-sum-closest/description

Given an integer array, find a subarray with sum closest to zero.

https://blog.csdn.net/willshine19/article/details/48494129

## Fast Power

Medium https://www.lintcode.com/problem/fast-power/description

Calculate the `a ^ n % b` where `a`, `b` and `n` are all 32bit non-negative integers.

https://blog.csdn.net/willshine19/article/details/48493253

## Sort Letters by Case

Medium https://www.lintcode.com/problem/sort-letters-by-case/

Given a string which contains only letters. Sort it by lower case first and upper case second.

https://blog.csdn.net/willshine19/article/details/48622531


---

# 滑动窗口

滑动窗口题目|
---|---
3. 无重复字符的最长子串 |
424. 替换后的最长重复字符 |
209. 长度最小的子数组 |
567. 字符串的排列 |
76. 最小覆盖子串 |

undo|
---|---
159. 至多包含两个不同字符的最长子串 |
30. 串联所有单词的子串 |
239. 滑动窗口最大值 |
632. 最小区间 |
727. 最小窗口子序列 |


3. 无重复字符的最长子串

https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/


```python
'''
思路: left right 双指针, 指向窗口的第一个元素和最后一个元素
'''
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s: return 0
        n = len(s)
        l = r = max_len = 0
        lookup = set()

        for r in range(n):
            while s[r] in lookup:
                lookup.remove(s[l])
                l += 1
            lookup.add(s[r])
            max_len = max(max_len, r - l + 1)
            
        return max_len
```

424. 替换后的最长重复字符

https://leetcode-cn.com/problems/longest-repeating-character-replacement/


```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        if not s: return 0
        from collections import Counter
        lookup = Counter() 
        n = len(s)
        l = r = max_len = 0

        for r in range(n):
            lookup[s[r]] += 1
            while sum(lookup.values()) - max(lookup.values()) > k:
                lookup[s[l]] -= 1
                l += 1
            max_len = max(max_len, r - l + 1)

        return max_len
```

209. 长度最小的子数组

https://leetcode-cn.com/problems/minimum-size-subarray-sum/

```python
'''
2020.2.8
时间 O(n)
'''
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums: return 0
        l = r = 0
        n = len(nums)
        min_len = n + 1
        window = 0
        
        for r in range(n):
            window += nums[r]
            while window >= s:
                min_len = min(min_len, r - l + 1)
                window -= nums[l]
                l += 1
        
        return 0 if min_len > n else min_len
```

567. 字符串的排列

https://leetcode-cn.com/problems/permutation-in-string/

```python
'''
2020.2.18
时间 O(nm)
'''
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        from collections import Counter
        n1, n2 = len(s1), len(s2)
        if n1 > n2: return False
        c1 = Counter(s1)
        window = Counter()

        for i, v in enumerate(s2):
            window[v] += 1
            drop_i = i - n1
            if drop_i >= 0:
                drop_v = s2[drop_i]
                window[drop_v] -= 1

            for k, v in c1.items():
                if k not in window: break
                if v != window[k]: break
            else:
                return True
        return False

```

76. 最小覆盖子串

https://leetcode-cn.com/problems/minimum-window-substring/

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t: return ""
        from collections import Counter
        lookup = Counter(t)
        window = Counter()

        def containsT(w):
            for k, v in lookup.items():
                if k not in w: return False
                if v > w[k]: return False
            else:
                return True

        N = len(s)
        ans = (N + 1, 0, 0)
        l = r = 0

        while r < N:
            if s[r] in lookup:
                window[s[r]] += 1
            while containsT(window):
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)
                if s[l] in lookup:
                    window[s[l]] -= 1
                l += 1
            r += 1
        if ans[0] == N + 1: return ""
        return s[ans[1]:ans[2] + 1]
```

239. 滑动窗口最大值

https://leetcode-cn.com/problems/sliding-window-maximum/


480. 滑动窗口中位数

https://leetcode-cn.com/problems/sliding-window-median/



268. 缺失数字

https://leetcode-cn.com/problems/missing-number/

方法一

map
时间 O(n)
空间 O(n)

方法二 

位运算
时间 O(n)
空间 O(1)

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        N = len(nums)
        ans = 0
        for num in nums:
            ans ^= num
        for num in range(N + 1):
            ans ^= num
        return ans
```


204. 计数质数

https://leetcode-cn.com/problems/count-primes/

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        if n < 2: return 0
        arr = [True] * n
        arr[0] = arr[1] = False
        i = 2
        while i * i < n:
            if arr[i]:
                j = 2
                while i * j < n:
                    arr[i * j] = False
                    j += 1
            i += 1
        count = 0
        for i in range(1, n):
            if arr[i]: count += 1
        return count

```

就是先先干掉2的倍数，然后干掉3的倍数，（4是2的倍数，过滤），干掉5的倍数，（6是2和3的倍数），干掉7的倍数，就这样子一路干到sqrt(n)就行了

![](https://pic.leetcode-cn.com/77583e8c9a820e3880f754a00863d616642d8cf915230382c2aaa11168c25849-file_1581643684036)

896. 单调数列

https://leetcode-cn.com/problems/monotonic-array/

```python
class Solution:
    def isMonotonic(self, A: List[int]) -> bool:
        if not A or len(A) == 1: return True
        mx = float("-inf")
        mn = float("inf")
        for i in range(1, len(A)):
            diff = A[i] - A[i - 1]
            mx = max(mx, diff)
            mn = min(mn, diff)
        return bool(mx * mn >= 0)
```