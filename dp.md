# 动态规划


blog

https://blog.csdn.net/willshine19/category_3219837.html

[toc]


## 96. 不同的二叉搜索树

https://leetcode-cn.com/problems/unique-binary-search-trees/

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


## 139. 单词拆分

https://leetcode-cn.com/problems/word-break/


回溯

dp

```python
'''
2020.2.22
思路: dp[i] 表示以 s[i] 为结尾的字符串, 是否可以
时间: O(n2), 假设在字典中查找的时间 O(1)
'''
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False] * n
        for i in range(n):
            if s[:i + 1] in wordDict: 
                dp[i] = True
                continue
            for left in range(i):
                if dp[left] and s[left + 1:i + 1] in wordDict:
                    dp[i] = True
                    break

        return dp[-1]

```

## 221. 最大正方形

https://leetcode-cn.com/problems/maximal-square/

暴力

dp

```python
'''
2020.2.22
思路 dp[r][c] 表示以它为右下角的最大正方形的边长
时间 O(n2)
空间 O(n2)
'''
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix: return 0
        n_row = len(matrix)
        if not n_row: return 0
        n_col = len(matrix[0])
        res = 0

        dp = [[0] * n_col for _ in range(n_row)]        
        for r in range(n_row):
            dp[r][0] = int(matrix[r][0])
        for c in range(n_col):
            dp[0][c] = int(matrix[0][c])

        for r in range(1, n_row):
            for c in range(1, n_col):
                if matrix[r][c] == "1": 
                    dp[r][c] = min(dp[r-1][c-1], dp[r-1][c], dp[r][c-1]) + 1

        side = max([max(row) for row in dp])
        return side * side
```

![](https://pic.leetcode-cn.com/28657155fcebc3f210982e889ceef89f6295fb48999222bfe0e52514158c446e-image.png)


## 279. 完全平方数

https://leetcode-cn.com/problems/perfect-squares/

bfs

dp

```python
'''
dp
dp[i] 表示 n = i 的结果
时间 O(n2)
空间 O(n)
'''
class Solution:
    def numSquares(self, n: int) -> int:
        dp = list(range(n + 1))
        for i in range(1, n + 1):
            for j in range(1, n):
                pre = i - j * j
                if pre < 0: break
                dp[i] = min(dp[i], dp[pre] + 1)
        return dp[-1]
```

## 70. 爬楼梯

https://leetcode-cn.com/problems/climbing-stairs/


```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        if n == 2: return 2
        dp = [1, 2] + [0] * (n - 2)
        for i in range(2, n):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]
```

```java

public class Solution {
    /**
     * @param n: An integer
     * @return: An integer
     */
    public int climbStairs(int n) {
        // 2015-05-14
        if (n == 0 || n == 1 || n == 2) {
            return n;
        }
        int[] sum = new int[n];
        sum[0] = 1;
        sum[1] = 2;
        for (int i = 2; i < n; i++) {
            sum[i] = sum[i - 1] + sum[i - 2];
        }
        return sum[n - 1];
    }
}
```

## 64. 最小路径和

https://leetcode-cn.com/problems/minimum-path-sum/

```python
class Solution:
    def minPathSum(self, grid: [[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == j == 0: continue
                elif i == 0:  grid[i][j] = grid[i][j - 1] + grid[i][j]
                elif j == 0:  grid[i][j] = grid[i - 1][j] + grid[i][j]
                else: grid[i][j] = min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j]
        return grid[-1][-1]
```

```java
public class Solution {
    /**
     * @param grid: a list of lists of integers.
     * @return: An integer, minimizes the sum of all numbers along its path
     */
    public int minPathSum(int[][] grid) {
        // 2015-05-13
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        
        int m = grid.length;
        int n = grid[0].length;
        int[][] sum = new int[m][n];
        
        sum[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            sum[i][0] = sum[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < n; j++) {
            sum[0][j] = sum[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                sum[i][j] = Math.min(sum[i - 1][j], sum[i][j - 1]) + grid[i][j];
            }
        }
        return sum[m - 1][n - 1];
    }
}
```

## 62. 不同路径

https://leetcode-cn.com/problems/unique-paths/

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/robot_maze.png)

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * m for _ in range(n)]
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = dp[i][j-1] + dp[i-1][j]
        return dp[-1][-1]
```


## 63. 不同路径 II

https://leetcode-cn.com/problems/unique-paths-ii/

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if not obstacleGrid: return 0
        n_r = len(obstacleGrid)
        if not obstacleGrid[0]: return 0
        n_c = len(obstacleGrid[0])

        dp = [[0] * n_c for _ in range(n_r)]
        dp[0][0] = 1 - obstacleGrid[0][0]
        for r in range(1, n_r):
            if obstacleGrid[r][0] == 0:
                dp[r][0] = dp[r - 1][0]
        for c in range(1, n_c):
            if obstacleGrid[0][c] == 0:
                dp[0][c] = dp[0][c - 1]
        for r in range(1, n_r):
            for c in range(1, n_c):
                if obstacleGrid[r][c] == 0: 
                    dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
        return dp[-1][-1]
```

## 121. 买卖股票的最佳时机

Easy https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/

only one transaction

方法1 O(n2)


n2

```kotlin
class Solution {
    fun maxProfit(prices: IntArray): Int {
        val n = prices.size
        var ret = 0
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                ret = max(ret, prices[j] - prices[i])
            }
        }
        return ret
    }
}
```


方法2 O(n)

```python
'''
2020.3.8
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0
        profit = 0
        price_min = prices[0]
        for price in prices:
            price_min = min(price_min, price)
            profit = max(profit, price - price_min)
        return profit
```

```java
public class Solution {
    public int maxProfit(int[] prices) {
        // 2015-09-15
        if (prices == null || prices.length == 0) {
            return 0;
        }
        
        int minPrice = Integer.MAX_VALUE;
        int rst = 0;
        for (int i = 0; i < prices.length; i++) {
            minPrice = Math.min(minPrice, prices[i]);
            rst = Math.max(rst, prices[i] - minPrice);
        }
        return rst;
    }
}
```

```kotlin
class Solution {
    fun maxProfit(prices: IntArray): Int {
        var lowPrice = Integer.MAX_VALUE
        var ret = 0
        for (i in 1 ..< prices.size) {
            lowPrice = min(lowPrice, prices[i - 1])
            ret = max(ret, prices[i] - lowPrice)
        }
        return ret
    }
}
```

## 122. 买卖股票的最佳时机 II

Easy https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/

as many transactions as you like

方法1 dfs

方法2 dp

```python
'''
0 表示 不持股, 第二天可以不持股或持股
1 表示 持股, 第二天可以持股或冻结
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0
        N = len(prices)
        dp = [[0] * N for _ in range(2)]
        dp[1][0] = -prices[0]
        for i in range(1, N):
            dp[0][i] = max(dp[0][i - 1], dp[1][i - 1] + prices[i])
            dp[1][i] = max(dp[0][i - 1] - prices[i], dp[1][i - 1])
        return dp[0][-1]
```

方法3 O(n) 贪心

```kotlin
class Solution {
    fun maxProfit(prices: IntArray): Int {
        var ret = 0
        for (i in 1 until prices.size) {
            ret += max(0, prices[i] - prices[i-1])
        }
        return ret
    }
}
```

```kotlin
class Solution {
    fun maxProfit(prices: IntArray): Int {
        return (1 ..< prices.size).sumOf { max(0, prices[it] - prices[it - 1])}
    }
}
```

```python
'''
2020.3.8
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0
        profit = 0
        for i in range(1, len(prices)):
            if prices[i - 1] < prices[i]:
                profit += prices[i] - prices[i - 1]
        return profit 
```

```java
class Solution {
    public int maxProfit(int[] prices) {
        // 2015-09-15
        if (prices == null || prices.length == 0) {
            return 0;
        }
        
        int rst = 0;
        for (int i = 1; i < prices.length; i++) {
            int sub = prices[i] - prices[i - 1];
            if (sub > 0) {
                rst += sub;
            }
        }
        return rst;
    }
};
```


## 714. 买卖股票的最佳时机含手续费

https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/

方法1 dp O(n)


```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        if not prices: return 0
        N = len(prices)
        dp = [[0] * N for _ in range(2)]
        dp[1][0] = -prices[0]
        for i in range(1, N):
            dp[0][i] = max(dp[0][i - 1], dp[1][i - 1] + prices[i] - fee)
            dp[1][i] = max(dp[0][i - 1] - prices[i], dp[1][i - 1])
        return dp[0][-1]
```

## 309. 最佳买卖股票时机含冷冻期

https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/

方法1 dp O(n)

```python
'''
0 表示 不持股, 第二天可以不持股或持股
1 表示 持股, 第二天可以持股或冻结
2 表示 冻结期, 第二天不持股
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0
        N = len(prices)
        dp = [[0] * N for _ in range(3)]
        dp[1][0] = -prices[0]
        for i in range(1, N):
            dp[0][i] = max(dp[2][i - 1], dp[0][i - 1])
            dp[1][i] = max(dp[0][i - 1] - prices[i], dp[1][i - 1])
            dp[2][i] = dp[1][i - 1] + prices[i]
        return max(dp[0][-1], dp[2][-1])
```

## 123. 买卖股票的最佳时机 III

Hard https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/

at most two transactions

方法1 双 dp O(n)

```python
'''
2020.3.8
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0

        left = []
        profit = 0
        price_min = prices[0]
        
        for price in prices:
            price_min = min(price_min, price)
            profit = max(profit, price - price_min)
            left.append(profit)

        right = []
        profit = 0
        price_max = prices[-1]
        for price in reversed(prices):
            price_max = max(price_max, price)
            profit = max(profit, price_max - price)
            right[0:0] = [profit]
        
        profit_max = 0
        for i in range(len(prices)):
            profit_max = max(profit_max, left[i] + right[i])
        return profit_max
```

```java
class Solution {
    public int maxProfit(int[] prices) {
        // 2015-09-15 dp
        if (prices == null || prices.length <= 1) {
            return 0;
        }
        
        // dp from left
        int[] left = new int[prices.length];
        int minPrice = Integer.MAX_VALUE;
        left[0] = 0;
        minPrice = prices[0];
        for (int i = 1; i < prices.length; i++) {
            minPrice = Math.min(minPrice, prices[i]);
            left[i] = Math.max(left[i - 1], prices[i] - minPrice);
        }
        
        // dp from right 
        int[] right = new int[prices.length];
        int maxPrice = Integer.MIN_VALUE;
        right[prices.length - 1] = 0;
        maxPrice = prices[prices.length - 1];
        for (int i = prices.length - 2; i >= 0; i--) {
            maxPrice = Math.max(maxPrice, prices[i]);
            right[i] = Math.max(right[i + 1], maxPrice - prices[i]);
        }
        
        int rst = 0;
        for (int i = 0; i < prices.length; i++) {
            rst = Math.max(rst, left[i] + right[i]);
        }
        return rst;
    }
};
```

Input: [3,3,5,0,0,3,1,4]
left:  [0,0,2,2,2,3,3,4]
right: [4,4,4,4,4,3,3,0]

## 188. 买卖股票的最佳时机 IV

https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/


## 72. 编辑距离

https://leetcode-cn.com/problems/edit-distance/

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        if n1 == 0: return n2
        if n2 == 0: return n1
        # n1 行 n2 列
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for r in range(n1 + 1):
            dp[r][0] = r
        for c in range(n2 + 1):
            dp[0][c] = c
        for r in range(1, n1 + 1):
            for c in range(1, n2 + 1):
                if word1[r - 1] == word2[c - 1]:
                    dp[r][c] = dp[r - 1][c - 1]
                else:
                    dp[r][c] = min(dp[r - 1][c], dp[r][c - 1], dp[r - 1][c - 1]) + 1
        return dp[-1][-1]
```