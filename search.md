# 回溯 backtrack

![](https://pic.leetcode-cn.com/6a464ba95a7ad1c247aa39610535984c241e6b95148f8bc36b02908a190b1d54-image.png)


题目 | 提示
--|--
46. 全排列 | [1,2,3] 给出所有排列
47. 全排列 II | [1,2,2] 给出所有排列, 为什么造成了重复，如何在搜索之前就判断这一支会产生重复，从而“剪枝”。
78. 子集 | [1,2,3] 给出所有子集, 为数不多的，解不在叶子结点上的回溯搜索问题。解法比较多，注意对比。
90. 子集 II | [1,2,2] 给出所有子集 剪枝技巧同 47 题、39 题、40 题。
17. 电话号码的字母组合 |
22. 括号生成 | 这是字符串问题，没有显式回溯的过程。这道题广度优先遍历也很好写，可以通过这个问题理解一下为什么回溯算法都是深度优先遍历，并且都用递归来写。
39. 组合总和 | 使用题目给的示例，画图分析。
40. 组合总和 II | 
51. N皇后 | 其实就是全排列问题，注意设计清楚状态变量。
60. 第k个排列 | 利用了剪枝的思想，减去了大量枝叶，直接来到需要的叶子结点。
77. 组合 | 组合问题按顺序找，就不会重复。并且举一个中等规模的例子，找到如何剪枝，这道题思想不难，难在编码。
79. 单词搜索 |
93. 复原IP地址 | 	
784. 字母大小写全排列 |	

## 题目

https://blog.csdn.net/willshine19/category_5631635.html


### 46 全排列 Permutations

https://leetcode-cn.com/problems/permutations/

给定一个没有重复数字的序列，返回其所有可能的全排列。

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

PY 1

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
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


78. 子集 Subsets

https://leetcode-cn.com/problems/subsets/



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



90. 子集 II Subsets II

https://leetcode-cn.com/problems/subsets-ii/

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