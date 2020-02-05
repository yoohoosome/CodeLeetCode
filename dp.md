

https://leetcode.com/problems/unique-binary-search-trees/

```python
class Solution:
    def numTrees(self, n: int) -> int:
        if n <= 1:
            return 1
        dp = [1, 1]
        for i in range(2, n + 1):
            dp += [0]
            for left in range(0, i):
                right = i - 1 - left
                dp[-1] +=  dp[left] * dp[right]
        return dp[-1]     
```