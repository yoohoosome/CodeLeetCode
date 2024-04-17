# leetcode

## 2020.3.2


123

96


11. 盛最多水的容器

https://leetcode-cn.com/problems/container-with-most-water/

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        if not height: return 0
        left = 0
        right = len(height) - 1
        ans = 0
        while left < right:
            ans = max(ans, (right - left) * min(height[left], height[right]))
            if height[left] < height[right]:
                left += 1
            else: 
                right -= 1
        return ans
```

375

368

994. 腐烂的橘子

392. 判断子序列

双指针

```python
class Solution(object):
    def isSubsequence(self, s, t):    
        i, j = 0, 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                j += 1
        return True if i == len(s) else False
```

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        if not s: return True
        if not t: return False
        i = 0
        N = len(t)
        for c in s:
            if i == N: return False
            while t[i] != c:
                i += 1
                if i == N: return False
            i += 1
        else:
            return True
```

415. 字符串相加

https://leetcode-cn.com/problems/add-strings/

49. 字母异位词分组

https://leetcode-cn.com/problems/group-anagrams/

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = collections.defaultdict(list)
        for word in strs:
            count = [0] * 26
            for c in word:
                count[ord(c) - ord("a")] += 1
            ans[tuple(count)].append(word)
        return list(ans.values())
```
215. 数组中的第K个最大元素

https://leetcode-cn.com/problems/kth-largest-element-in-an-array/

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k, nums)[-1]
```