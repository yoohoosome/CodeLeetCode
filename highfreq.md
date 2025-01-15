# High Frequency

[toc]

## O(1) Check Power of 2

https://www.lintcode.com/problem/o1-check-power-of-2/description

```java
class Solution {
    /*
     * @param n: An integer
     * @return: True or false
     */
    public boolean checkPowerOf2(int n) {
        // 2015-09-06
        if (n <= 0) {
            return false;
        }
        boolean res = ((n & (n-1)) == 0) ? true : false;
        return res;
    }
};
```


## Sqrt(x)

https://www.lintcode.com/problem/sqrtx/description

```java
class Solution {
    /**
     * @param x: An integer
     * @return: The sqrt of x
     */
    public int sqrt(int x) {
        // 2015-09-06 用int会溢出
        long start = 0;
        long end = x / 2 + 1;
        while (start <= end) {
            long mid = start + (end - start) / 2;
            if ((mid * mid) > x) {
                end = mid - 1;
            } else if ((mid + 1) * (mid + 1) <= x) {
                start = mid + 1;
            } else {
                return (int)mid;
            }
        }
        return 0;
    }
}
```


162
153


264
204


134
738


86
15

75.颜色分类
https://leetcode-cn.com/problems/sort-colors/

973. 最接近原点的K个数
https://leetcode-cn.com/problems/k-closest-points-to-origin/

22. Generate Parentheses
51. N-Queens （选做52. N-Queens II）


215 数组中第k个最大元素
https://leetcode-cn.com/problems/kth-largest-element-in-an-array/

240 搜索二维矩阵ii
https://leetcode-cn.com/problems/search-a-2d-matrix/
https://leetcode-cn.com/problems/search-a-2d-matrix-ii/


```python
'''
71. 简化路径
https://leetcode-cn.com/problems/simplify-path/
思路: 栈
特殊路径: . .. // 
特殊返回: 根目录 
'''
class Solution:
    def simplifyPath(self, path: str) -> str:
        path_stack = []
        dirs = path.split('/')
        for e in dirs:
            if not e:
                continue 
            if e == "..":
                if path_stack: path_stack.pop()
            elif e != ".":
                path_stack.append(e)
        simple = ""
        for e in path_stack:
            simple = simple + "/" + e
        return simple if simple else "/"
```


```python
'''
145. 二叉树的后序遍历
https://leetcode-cn.com/problems/binary-tree-postorder-traversal/
'''
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]
```


https://leetcode-cn.com/problems/number-of-islands/

```python
'''
200-岛屿数量
https://leetcode-cn.com/problems/number-of-islands/
思路：深度优先搜索，染色标记
难点: dfs 写法, 矩阵
时间复杂度：O(M*N)
空间复杂度：O(M*N)
'''
class Solution:
    def dfs(self, grid, r, c):
        nr = len(grid)
        nc = len(grid[0])
        if r < 0 or c < 0 or r >= nr or c >= nc:
            return
        if grid[r][c] == '0':
            return
        grid[r][c] = '0'
        self.dfs(grid, r + 1, c)
        self.dfs(grid, r - 1, c)
        self.dfs(grid, r, c + 1)
        self.dfs(grid, r, c - 1)

    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not len(grid) or not len(grid[0]):
            return 0
        res = 0
        nr = len(grid)
        nc = len(grid[0])
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == '1':
                    res += 1
                    self.dfs(grid, r, c)
        return res
```


```python
'''
128. 最长连续序列 
https://leetcode-cn.com/problems/longest-consecutive-sequence/
哈希表
时间复杂度：O(n) while循环只在num为第一个才执行，且执行时间不会超过n
空间复杂度：O(n)
'''
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0
        longest = 0
        nums_set = set(nums)
        for num in nums_set:
            if (num - 1) in nums_set:
                continue
            length = 1
            while num + 1 in nums_set:
                length += 1
                num += 1
            longest = max(longest, length)
        return longest
```

## 2020.2.1

```python
'''
395. 至少有K个重复字符的最长子串
https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters
分治: 分割成相同的子问题
分割方法: 当一个字符出现的次数小于k时，该字符肯定不会出现在某个满足条件的子串中，即该字符可以把字符串分割
'''
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        if not s:
            return 0
        for c in set(s):
            if s.count(c) < k:
                return max(self.longestSubstring(sub, k) for sub in s.split(c))
        else:
            return len(s)
```

```python
'''
718. 最长重复子数组
https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/
动态规划
二维数组: dp[i][j] 为 A[i:] 和 B[j:] 的最长公共前缀, return max(dp)
状态转移: dp[i][j] = dp[i + 1][j + 1] + 1 if A[i] == B[j] else 0
'''
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        dp = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
        for i in range(len(A) - 1, -1, -1):
            for j in range(len(B) - 1, -1, -1):
                if A[i] == B[j]:
                    dp[i][j] = dp[i+1][j+1]+1
        return max(max(r) for r in dp)
```

## 2020.2.2 

```python
'''
46. 全排列
https://leetcode-cn.com/problems/permutations/
回溯搜索 dfs
'''

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums: return []
        if len(nums) == 1: return [nums]
        res = []
        for i in range(len(nums)):
            for rest in self.permute(nums[:i] + nums[i+1:]):
                res.append([nums[i]] + rest)
        return res
```

```python
'''
47. 全排列 II
https://leetcode-cn.com/problems/permutations-ii/
排序 + dfs
'''
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        if len(nums) == 1: return [nums]
        nums.sort()
        res = []
        for i in range(len(nums)):
            if i > 0 and nums[i - 1] == nums[i]: continue
            for rest in self.permuteUnique(nums[:i] + nums[i+1:]):
                res.append([nums[i]] + rest)
        return res
```

```python
'''
79. 单词搜索
https://leetcode-cn.com/problems/word-search/
dfs
'''
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        nr = len(board)
        if nr == 0: return false
        nc = len(board[0])
        used = [[False for _ in range(nc)] for __ in range(nr)]
        for r in range(nr):
            for c in range(nc):
                if self.backtrack(board, used, word, 0, r, c):
                    return True
        return False
    
    def backtrack(self, board, used, word, index, r, c) -> bool:
        if index == len(word): return True
        nr = len(board)
        nc = len(board[0])
        if r < 0 or r >= nr: return False
        if c < 0 or c >= nc: return False
        if used[r][c]:
            return False
        if board[r][c] == word[index]:
            used[r][c] = True
            if self.backtrack(board, used, word, index + 1, r + 1, c): return True 
            if self.backtrack(board, used, word, index + 1, r - 1, c): return True 
            if self.backtrack(board, used, word, index + 1, r, c + 1): return True 
            if self.backtrack(board, used, word, index + 1, r, c - 1): return True
        used[r][c] = False 
        return False
```

## 2020.2.3

```python
'''
55. 跳跃游戏
https://leetcode-cn.com/problems/jump-game/
dp
时间 O(n*n)
'''
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if not nums: return False
        n = len(nums)
        dp = [False for _ in range(n)]
        dp[0] = True
        for i in range(1, n):
            for prev in range(i - 1, -1, -1):
                if dp[prev] and prev + nums[prev] >= i: 
                    dp[i] = True
                    break
        return dp[-1]
```


```python
'''
55. 跳跃游戏
https://leetcode-cn.com/problems/jump-game/
贪心
如果一个位置能够到达，那么这个位置左侧所有位置都能到达
时间 O(n)
空间 O(1)
'''
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if not nums: return False
        n = len(nums)
        longest = 0
        for i in range(n):
            if i > longest:
                # 最远到longest, 无法再跳
                return False
            longest = max(longest, nums[i] + i)
        return True
```

```python
'''
300. 最长上升子序列
https://leetcode-cn.com/problems/longest-increasing-subsequence/
dp
时间 O(n*n)
空间 O(n)
'''
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums: return 0
        dp = [0] * len(nums)
        for right in range(0, len(nums)):
            m = 1
            for left in range(0, right):
                if nums[left] < nums[right]:
                    m = max(m, dp[left] + 1)
            dp[right] = m
        return max(dp)
```


648. 单词替换

https://leetcode-cn.com/problems/replace-words/

## 2020.2.17

34. 在排序数组中查找元素的第一个和最后一个位置

https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/

4. 寻找两个有序数组的中位数

https://leetcode-cn.com/problems/median-of-two-sorted-arrays/




## 2020.2.24


20. 有效的括号

https://leetcode-cn.com/problems/valid-parentheses/

```python
'''
2020.2.24
stack
'''
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for c in s:
            if c in "{[(": 
                stack.append(c)
                continue
            if c in ")]}" and not stack: return False
            if c == ")" and stack.pop() != "(": return False
            if c == "]" and stack.pop() != "[": return False
            if c == "}" and stack.pop() != "{": return False
        return True if not stack else False
```

150. 逆波兰表达式求值

https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/

```python
'''
2020.2.24
stack
'''
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = list()
        for t in tokens:
            if t not in '+-*/':
                stack.append(int(t))
            else:
                right = stack.pop()
                left = stack.pop()
                if t == '+':
                    stack.append(left + right)
                elif t == '-':
                    stack.append(left - right)
                elif t == '*':
                    stack.append(left * right)
                elif t == '/':
                    stack.append(int(left / right))
        # 6 // -123 = -1?
        return stack[0]
```

https://leetcode-cn.com/problems/longest-palindromic-substring/


5. 最长回文子串

暴力 O(n3)

dp O(n2)

![](https://pic.leetcode-cn.com/e8c42f244ff5cb91096dfbdb70d2ccefdb88d428a111d89f4e3ac42dcd337103-image.png)




31. 下一个排列

https://leetcode-cn.com/problems/next-permutation/

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        def reverse(nums, left, right):
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        
        n = len(nums)
        if n <= 1: return
        idx = n - 2
        while idx >= 0 and nums[idx] >= nums[idx + 1]:
            idx -= 1
        if idx < 0: 
            reverse(nums, 0, n - 1)
            return
        left = idx
        idx = n - 1
        while nums[left] >= nums[idx]:
            idx -= 1
        right = idx
        nums[left], nums[right] = nums[right], nums[left]
        reverse(nums, left + 1,  n - 1)

```
49. 字母异位词分组

https://leetcode-cn.com/problems/group-anagrams/

```python
class Solution:
    def groupAnagrams(strs):
        ans = collections.defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(s)
        return ans.values()
```

67. 二进制求和

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        if not a or int(a) == 0: return b
        if not b or int(b) == 0: return a
        n1 = len(a)
        n2 = len(b)
        if n1 < n2:
            a = "0" * (n2 - n1) + a
        elif n2 < n1:
            b = "0" * (n1 - n2) + b
        n = max(n1, n2)
        carry = 0
        ans = []
        for i in range(n - 1, -1, -1):
            s = int(a[i]) + int(b[i]) + carry
            carry = s // 2
            ans = [str(s % 2)] + ans
        if carry == 1: ans = ["1"] + ans
        return "".join(ans)
```

647. 回文子串

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        ans = 0
        for center in range(2 * n - 1):
            left = center // 2
            right = left + center % 2
            while left >= 0 and right < n and s[left] == s[right]:
                ans += 1
                left -= 1
                right += 1
        return ans

```

560. 和为K的子数组


```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        from collections import Counter
        ans = 0
        s = 0
        lookup = Counter()
        for v in nums:
            lookup[s] += 1
            s += v
            if (s - k) in lookup:
                ans += lookup[s - k]
        return ans
```

225. 用队列实现栈

https://leetcode-cn.com/problems/implement-stack-using-queues/

```python
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.t = []


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.t.append(x)


    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.t.pop()


    def top(self) -> int:
        """
        Get the top element.
        """
        return self.t[-1]


    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return not bool(self.t)
```