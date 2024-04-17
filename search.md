# 回溯 backtrack

[toc]

![](https://pic.leetcode-cn.com/6a464ba95a7ad1c247aa39610535984c241e6b95148f8bc36b02908a190b1d54-image.png)

解题要点

1. 画树
1. 每个节点包括 (1) 当前解 (2) 可用元素

https://leetcode-cn.com/problems/permutations/solution/hui-su-suan-fa-python-dai-ma-java-dai-ma-by-liweiw/


题目 | 提示
--|--
[46. 全排列](https://leetcode-cn.com/problems/permutations/) | [1,2,3] 给出所有排列
[47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/) | [1,2,2] 给出所有排列, 为什么造成了重复，如何在搜索之前就判断这一支会产生重复，从而“剪枝”。
[78. 子集](https://leetcode-cn.com/problems/subsets/) | [1,2,3] 给出所有子集 (解不在叶子结点上的回溯搜索问题)
[90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/) | [1,2,2] 给出所有子集 剪枝技巧同 47 题、39 题、40 题。
[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/) | candidates = [2,3,6,7], target = 7, 数字可以用多次
[40. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/) | candidates = [10,1,2,7,6,1,5], target = 8, 数字只能用一次
[51. N皇后](https://leetcode-cn.com/problems/n-queens/) | 其实就是全排列问题，注意设计清楚状态变量。
[52. N皇后 II](https://leetcode-cn.com/problems/n-queens-ii/) |
[131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning) |
[17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/) |
[22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/) | 这是字符串问题，没有显式回溯的过程。这道题广度优先遍历也很好写，可以通过这个问题理解一下为什么回溯算法都是深度优先遍历，并且都用递归来写。
[60. 第k个排列](https://leetcode-cn.com/problems/permutation-sequence/) | 利用了剪枝的思想，减去了大量枝叶，直接来到需要的叶子结点。
[77. 组合](https://leetcode-cn.com/problems/combinations/) | 组合问题按顺序找，就不会重复。并且举一个中等规模的例子，找到如何剪枝，这道题思想不难，难在编码。
[79. 单词搜索](https://leetcode-cn.com/problems/word-search/) |
[93. 复原IP地址](https://leetcode-cn.com/problems/restore-ip-addresses/) | 	
[784. 字母大小写全排列](https://leetcode-cn.com/problems/letter-case-permutation/) |	
127. Word Ladder |	

## 题目

https://blog.csdn.net/willshine19/category_5631635.html


### 46 全排列 Permutations

https://leetcode-cn.com/problems/permutations/

给定一个没有重复数字的序列，返回其所有可能的全排列。


PY 1

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        # 2020.2.10
        if not nums: return [[]]
        res = []

        def helper(lst, ava):
            if not ava: 
                res.append(lst)
                return
                
            for i, v in enumerate(ava):
                helper(lst + [v], ava[:i] + ava[i+1:])
        
        helper([], nums)
        return res

```


```python
'''
46. 全排列
https://leetcode-cn.com/problems/permutations/
回溯搜索 分治?
'''

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums: return [[]]
        if len(nums) == 1: return [nums]
        # nums 保存未被使用的数

        res = []
        for i in range(len(nums)):
            for rest in self.permute(nums[:i] + nums[i+1:]):
                res.append([nums[i]] + rest)
        return res
```

```python
'''
2020.2.7
回溯搜索 dfs
'''
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums: return [[]]
        res = []


        def helper(lst):
            # lst 保存已经使用的数
            if len(lst) == len(nums):
                res.append(lst)
                return
            for v in nums:
                if v not in lst: helper(lst + [v])
        
        helper([])
        return res

```

### 47. 全排列 II Permutations II

https://leetcode-cn.com/problems/permutations-ii/

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # 2020.2.10
        if not nums: return [[]]
        nums.sort()
        res = []

        def helper(lst, ava):
            if not ava: res.append(lst)
            for i, v in enumerate(ava):
                if i > 0 and ava[i - 1] == ava[i]: continue
                helper(lst + [v], ava[:i] + ava[i+1:])
        helper([], nums)
        return res
```


```python
'''
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

```java
class Solution {
    /**
     * @param nums: A list of integers.
     * @return: A list of unique permutations.
     */
    public ArrayList<ArrayList<Integer>> permuteUnique(ArrayList<Integer> nums) {
        // 2015-08-28
        // nums可以含重复元，每个元素只能使用一次
        // 解集中不可以有重复的解
        ArrayList<ArrayList<Integer>> rst = new ArrayList<>();
        if (nums == null || nums.size() == 0) {
            return rst;
        }
        Collections.sort(nums);
        ArrayList<Integer> list = new ArrayList<>();
        
        // 关键：加了一个数组，用它来记录哪个元素已经用过了
        int[] visited = new int[nums.size()]; 
        
        helper(nums, rst, list, visited);
        return rst;
    }
    
    private void helper(ArrayList<Integer> nums, ArrayList<ArrayList<Integer>> rst,
            ArrayList<Integer> list, int[] visited) {
        if (list.size() == nums.size()) {
            rst.add(new ArrayList<Integer>(list));
            return;
        }
        for (int i = 0; i < nums.size(); i++) {
            // 避免重复的解
            if (i != 0 && nums.get(i - 1) == nums.get(i) && visited[i - 1] == 0) {
                continue;
            }
            // 确保不会重复使用同一个元素
            if (visited[i] == 1) {
                continue;
            }
            visited[i] = 1;
            list.add(nums.get(i));
            helper(nums, rst, list, visited);
            visited[i] = 0;
            list.remove(list.size() - 1);
        }
        return;
    }
}
```


JAVA 1

```java
class Solution {
    /**
     * @param nums: A list of integers.
     * @return: A list of permutations.
     */
    public ArrayList<ArrayList<Integer>> permute(ArrayList<Integer> nums) {
        // 2015-08-28
        // nums不含重复元素，每个元素只可以使用一次
        // 解集中不可以含重复解
        ArrayList<ArrayList<Integer>> rst = new ArrayList<>();
        if (nums == null || nums.size() == 0) {
            return rst;
        }
        Collections.sort(nums);
        ArrayList<Integer> list = new ArrayList<>();
        helper(nums, rst, list);
        return rst;
    }
    
    private void helper(ArrayList<Integer> nums, 
            ArrayList<ArrayList<Integer> rst, ArrayList<Integer> list) {
        if (list.size() == nums.size()) {
            rst.add(new ArrayList<Integer>(list));
            return;
        }    
        
        for (int i = 0; i < nums.size(); i++) {
            if (list.contains(nums.get(i))) {
                continue;
            }
            list.add(nums.get(i));
            helper(nums, rst, list);
            list.remove(list.size() - 1);
        }
        return;
    }
}
```


### 78. 子集 Subsets

https://leetcode-cn.com/problems/subsets/

PY 1

```python
'''
2020.2.7
'''
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        if not nums: return []
        res = []

        def helper(lst, available):
            res.append(lst)

            for i, v in enumerate(available):
                helper(lst + [v], available[i + 1:])
        
        helper([], nums)
        return res
```

```python
'''
2020.2.6
'''
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        if not nums: return [[]]
        n = len(nums)
        res = []
        def helper(lst, pos):
            res.append(lst)
            for i in range(pos, n):
                # lst + [nums[i]] 的目的是创建一个新的 list
                helper(lst + [nums[i]], i + 1)
        helper([], 0)
        return res
```

JAVA 1

```java
class Solution {
    public ArrayList<ArrayList<Integer>> subsets(ArrayList<Integer> S) {
        // 2015-07-07
        ArrayList<ArrayList<Integer>> rst = new ArrayList<>();
        if (S == null) {
            return rst;
        }
        ArrayList<Integer> list = new ArrayList<>();
        
        helper(rst, list, S);
        return rst;
    }
    
    private void helper(ArrayList<ArrayList<Integer>> rst, 
            ArrayList<Integer>list, ArrayList<Integer> S) {
        rst.add(new ArrayList<Integer>(list));
        // 树的深度
        // if (list.size() == S.size()) {
        //     return;
        // }
        
        for (int i = 0; i < S.size(); i++) {
            if (list.size() > 0 && list.get(list.size() - 1) >= S.get(i)) {
                continue;
            }
            list.add(S.get(i));
            helper(rst, list, S);
            list.remove(list.size() - 1);
        }
        return;
    }
}

```

```java
class Solution {
    public ArrayList<ArrayList<Integer>> subsets(ArrayList<Integer> S) {
        // 2015-08-28
        // S中不含相同元素，每个元素只可以用一次
        // 解集中不可以含相同解
        // 因此 解中不会含相同元素
        ArrayList<ArrayList<Integer>> rst = new ArrayList<>();
        if (S == null || S.size() == 0) {
            return rst;
        }
        Collections.sort(S);
        ArrayList<Integer> list = new ArrayList<>();
        helper(S, rst, list, 0);
        return rst;
        
    }
    
    private void helper(ArrayList<Integer> S, ArrayList<ArrayList<Integer>> rst, 
            ArrayList<Integer> list, int pos) {
        rst.add(new ArrayList<Integer>(list));
        // if (pos == S.size()) {
        //     return;
        // }
        for (int i = pos; i < S.size(); i++) {
            list.add(S.get(i));
            helper(S, rst, list, i + 1);
            list.remove(list.size() - 1);
        }
    }
}
```

### 90. 子集 II Subsets II

https://leetcode-cn.com/problems/subsets-ii/

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        # 2020.2.12
        if not nums: return [[]]
        nums.sort()
        res = []

        def helper(lst, ava):
            res.append(lst)
            for i, v in enumerate(ava):
                if i > 0 and ava[i - 1] == ava[i]: continue
                helper(lst + [v], ava[i + 1:])
        
        helper([], nums)
        return res
```

```java
class Solution {
    /**
     * @param S: A set of numbers.
     * @return: A list of lists. All valid subsets.
     */
    public ArrayList<ArrayList<Integer>> subsetsWithDup(ArrayList<Integer> S) {
        // 2015-08-28
        // S中可能有重复元素，每个元素只可用一次
        // 解集中不可含重复的解
        // 解中可以含重复元素
        ArrayList<ArrayList<Integer>> rst = new ArrayList<>();
        if (S == null || S.size() == 0) {
            return rst;
        }
        Collections.sort(S);
        ArrayList<Integer> list = new ArrayList<>();
        helper(S, rst, list, 0);
        return rst;
    }
    
    private void helper(ArrayList<Integer> S, ArrayList<ArrayList<Integer>> rst, 
            ArrayList<Integer> list, int pos) {
        rst.add(new ArrayList<Integer>(list));
        // i从pos开始，不用再控制数的深度
        for (int i = pos; i < S.size(); i++) {
            // 控制树的分支 关键  
            if (i != pos && S.get(i - 1) == S.get(i)) {
                continue;
            }
            list.add(S.get(i));
            helper(S, rst, list, i + 1);
            list.remove(list.size() - 1);
        }
    }
}
```

### 39. 组合总和

https://leetcode-cn.com/problems/combination-sum/submissions/

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # 2020.2.10
        if not candidates: return [[]]
        candidates.sort()
        res = []

        def helper(lst, ava, sm):
            if sm == target: res.append(lst)
            if sm >= target: return
            for i, v in enumerate(ava):
                if sm + v > target: break
                helper(lst + [v], ava[i:], sm + v)

        helper([], candidates, 0)
        return res 
```

```java
public class Solution {
    /**
     * @param candidates: A list of integers
     * @param target:An integer
     * @return: A list of lists of integers
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // 2015-08-28
        // candicates中不含重复元素，元素可以重复使用
        // 解集中不可以含相同解
        List<List<Integer>> rst = new ArrayList<List<Integer>>();
        if (candidates == null || candidates.length == 0) {
            return rst;
        }
        
        List<Integer> list = new ArrayList<>();
        Arrays.sort(candidates);
        helper(candidates, rst, list, target, 0);
        return rst;     
    }
    
    private void helper(int[] candidates, List<List<Integer>> rst, List<Integer> list,
            int left, int pos) {
        if (left == 0) {
            rst.add(new ArrayList<Integer>(list));
            return;
        }
        
        for (int i = pos; i < candidates.length; i++) {
            if (left - candidates[i] < 0) {
                break;
            }
            list.add(candidates[i]);
            helper(candidates, rst, list, left - candidates[i], i);
            list.remove(list.size() - 1);
        }
    }
}
```

### 40. 组合总和 II

https://leetcode-cn.com/problems/combination-sum-ii/

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # 2020.1.10
        if not candidates: return [[]]
        candidates.sort()
        res = []
        
        def helper(lst, ava, sm):
            if sm == target: 
                res.append(lst)
                return
            for i, v in enumerate(ava):
                if v + sm > target: break
                if i > 0 and ava[i - 1] == ava[i]: continue
                helper(lst + [v], ava[i + 1:], sm + v)
        
        helper([], candidates, 0)
        return res
                
```

```java
public class Solution {
    /**
     * @param num: Given the candidate numbers
     * @param target: Given the target number
     * @return: All the combinations that sum to target
     */
    public List<List<Integer>> combinationSum2(int[] num, int target) {
        // 2015-07-09
        // num中可以含重复元素，每个元素只能使用一次
        // 解集中不可以含相同的解
        List<List<Integer>> rst = new ArrayList<List<Integer>>();
        if (num == null || num.length == 0) {
            return rst;
        }
        Arrays.sort(num);
        ArrayList<Integer> list = new ArrayList<>();
        helper(rst, list, num, target, 0);
        return rst;
    }
    
    private void helper(List<List<Integer>> rst, ArrayList<Integer> list, int[] num, int t, int pos) {
        if (t == 0) {
            rst.add(new ArrayList<Integer>(list));
            return;
        }
        for (int i = pos; i < num.length; i++) {
            if (t - num[i] < 0) {
                break;
            }
            if (i != pos && num[i] == num[i - 1]) { // 关键：注意是pos
                continue;
            }
            list.add(num[i]);
            helper(rst, list, num, t - num[i], i + 1);
            list.remove(list.size() - 1);
        }
        return;
    }
}
```

### N-Queens

51. N皇后
    
https://leetcode-cn.com/problems/n-queens/

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # 2020.2.11
        if not n: return [[]]
        res = []

        def isValid(lst, c):
            r = len(lst)
            for rq, cq in enumerate(lst):
                if cq == c: return False
                if cq + rq == c + r: return False
                if cq - rq == c - r: return False
            return True
        
        def toString(lst):
            queens = []
            for c in lst:
                queens.append('.' * c + 'Q' + '.' * (n - c - 1))
            return queens

        def helper(lst):
            if len(lst) == n: 
                res.append(toString(lst))
                return

            for c in range(n):
                if isValid(lst, c): helper(lst + [c])

        helper([])
        return res

```

```java
class Solution {
    /**
     * Get all distinct N-Queen solutions
     * @param n: The number of queens
     * @return: All distinct solutions
     * For example, A string '...Q' shows a queen on forth position
     */
    ArrayList<ArrayList<String>> solveNQueens(int n) {
        // 2015-07-07
        ArrayList<ArrayList<String>> rst = new ArrayList<>();
        if (n == 0) {
            return rst;
        }
        ArrayList<Integer> list = new ArrayList<>();
        helper(rst, list, n);
        return rst;
    }
    
    private void helper(ArrayList<ArrayList<String>> rst, ArrayList<Integer> list ,int n) {
        // 控制树的深度
        if (list.size() == n) {
            rst.add(convert(list));
            return;
        }
        
        for (int i = 0; i < n; i++) {
            // 控制分支
            if (!isValid(list, i)) {
                continue;
            }
            list.add(i);
            helper(rst, list, n);
            list.remove(list.size() - 1);
        }
        return;
    }
    
    // 将数组转为string组
    private ArrayList<String> convert(ArrayList<Integer> list) {
        ArrayList<String> rst = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            String s = new String("");
            for (int j = 0; j < list.size(); j++) {
                if (j == list.get(i)) {
                    s += "Q";
                } else {
                    s += ".";
                }
            }
            rst.add(s);
        }
        return rst;
    }
    
    // 判断是queen是否可以相互攻击
    private boolean isValid(ArrayList<Integer> list, int k) {
        if (list.size() == 0) {
            return true;
        }
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == k) {
                return false;
            }
            if (list.get(i) - i == k - list.size()) {
                return false;
            }
            if (list.get(i) + i == k + list.size()) {
                return false;
            }
        }
        return true;
    }
};
```


52. N皇后 II

https://leetcode-cn.com/problems/n-queens-ii/

```java
class Solution {
    /**
     * Calculate the total number of distinct N-Queen solutions.
     * @param n: The number of queens.
     * @return: The total number of distinct solutions.
     */
    public static int sum;
        
    public int totalNQueens(int n) {
        sum = 0;
        ArrayList<Integer> list = new ArrayList<>();
        helper(list, n);
        return sum;
    }
    
    /**
     * 递归方法
     * @param list : 已确定的queen的位置
     * @param n : queen的总个数
     */
    private void helper(ArrayList<Integer> list, int n) {  
        // 退出条件，控制树的深度  
        if (list.size() == n) {  
            sum++;  // 找到一个解
            return;  
        }  
          
        for (int i = 0; i < n; i++) {  
            // 控制分支  
            if (!isValid(list, i)) {  
                continue;  
            }  
            list.add(i);  
            helper(list, n);  
            list.remove(list.size() - 1);  
        }  
        return;  
    }  
    
    /**
     * 判断是queen是否可以相互攻击
     * @param list : 已确定的queen的位置
     * @param col : 新的一行中queen的位置
     */
    private boolean isValid(ArrayList<Integer> list, int col) {  
        if (list.size() == 0) {  
            return true;  
        }  
        for (int i = 0; i < list.size(); i++) {  
            if (list.get(i) == col) {  
                return false;  
            }  
            // 列 - 行
            if (list.get(i) - i == col - list.size()) {  
                return false;  
            }  
            // 列 + 行
            if (list.get(i) + i == col + list.size()) {  
                return false;  
            }  
        }  
        return true;  
    }  
};
```

### 131. 分割回文串

https://leetcode-cn.com/problems/palindrome-partitioning/

https://blog.csdn.net/willshine19/article/details/46808075

```

   a a b
   |      |
[a] a b  [aa] b
   |      |
[a,a] b  [aa,b]
   |      
[a,a,b]

```

```python
class Solution:
    # 2020.2.11
    def partition(self, s: str) -> List[List[str]]:
        res = []

        def helper(lst, ava):
            if not ava: res.append(lst)

            for i, v in enumerate(ava):
                nxt = ava[:i + 1]
                if nxt == nxt[::-1]:
                    helper(lst + [nxt], ava[i + 1:])
        
        helper([], s)
        return res

```

```java
public class Solution {
    /**
     * @param s: A string
     * @return: A list of lists of string
     */
    public List<List<String>> partition(String s) {
        // 2015-07-08
        List<List<String>> rst = new ArrayList<List<String>>();
        if (s == null || s.length() == 0) {
            return rst;
        }
        ArrayList<String> list = new ArrayList<>();
        
        helper(rst, list, s, 0);
        return rst;
    }
    
    // 注意第一个参数的类型
    private void helper(List<List<String>> rst, ArrayList<String> list, String s, int pos) {
        if (pos == s.length()) {
            rst.add(new ArrayList<String>(list));
            return;
        }
        
        for (int i = pos + 1; i <= s.length(); i++) { //注意是<=
            if (!isPalindrome(s.substring(pos, i))) {
                continue;
            }
            list.add(s.substring(pos, i));
            helper(rst, list, s, i);
            list.remove(list.size() - 1);
        }
        return;
    }
    
    private boolean isPalindrome(String s) {
        if (s.length() == 0) {
            return true;
        }
        int start = 0;
        int end = s.length() - 1;
        while (start < end) {
            if (s.charAt(start) != s.charAt(end)) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }
}
```

### 17. 电话号码的字母组合

https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        # 2020.2.12
        mapping = {"2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz"}

        res = []
        def helper(s, pos):
            if pos == len(digits):
                if s: res.append(s)
                return

            for i, v in enumerate(mapping[digits[pos]]) :
                helper(s + v, pos + 1)

        helper("", 0)
        return res 
```

### 22. 括号生成

https://leetcode-cn.com/problems/generate-parentheses/

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        # 2020.2.12
        res = []

        def helper(s: str, left: int, right: int):
            if len(s) == n * 2: 
                res.append(s)
                return
            if left < n:
                helper(s + "(", left + 1, right)
            if right < left:
                helper(s + ")", left, right + 1)
        
        helper("", 0, 0)
        return res
```


### 77. 组合
    
https://leetcode-cn.com/problems/combinations/

### 79. 单词搜索

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

93. 复原IP地址

https://leetcode-cn.com/problems/restore-ip-addresses/

```python
'''
本题目有几个需要注意的小 case
'''
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        if not s: return []
        res = []

        def helper(lst, ava):
            if len(lst) == 4:
                if not ava: res.append(".".join(lst))
                return
            n = len(ava)
            for i in range(n):
                nxt = ava[:i + 1]
                if int(nxt) > 255: break 
                if nxt[0] == "0" and len(nxt) > 1: continue
                helper(lst + [nxt], ava[i + 1:])
        
        helper([], s)
        return res
```

784. 字母大小写全排列

https://leetcode-cn.com/problems/letter-case-permutation/

```python
class Solution:
    def letterCasePermutation(self, S: str) -> List[str]:
        if not S: return []
        res = []

        def helper(s, pos):
            if pos == len(S): 
                res.append(s)
                return
            if S[pos].isalpha():
                helper(s + S[pos].upper(), pos + 1)
                helper(s + S[pos].lower(), pos + 1)
            else:
                helper(s + S[pos], pos + 1)
        
        helper("", 0)
        return res
```

## todo

https://blog.csdn.net/willshine19/article/details/48129521




---

## 广度优先

### 127. Word Ladder

https://leetcode-cn.com/problems/word-ladder/

https://leetcode-cn.com/problems/word-ladder-ii/

```
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]
```

![](https://pic.leetcode-cn.com/fc3a60e60cb7a80723feea0689c25a6f1637df8c64cfec0d70a264eee7e88254-Word_Ladder_1.png)

https://blog.csdn.net/willshine19/article/details/46840853

https://blog.csdn.net/willshine19/article/details/48104265