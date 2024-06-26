# Tree

[toc]

## 层序遍历

### 102. 二叉树的层次遍历

Binary Tree Level Order Traversal

https://leetcode-cn.com/problems/binary-tree-level-order-traversal/

Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        rst, left = [], []
        if root: left.append(root)
        while left:
            level = []
            for i in range(len(left)):
                node = left.pop(0)
                level.append(node.val)
                if node.left: left.append(node.left)
                if node.right: left.append(node.right)
            rst.append(level)
        return rst
```

```java
public class Solution {
    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        // 2015-3-22 BFS
        ArrayList<ArrayList<Integer>> rst = new ArrayList<>();
        if (root == null) {
            return rst;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            ArrayList<Integer> level = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode temp = queue.poll();
                level.add(temp.val);
                if (temp.left != null) {
                    queue.offer(temp.left);
                }
                if (temp.right != null) {
                    queue.offer(temp.right);
                }
            }
            rst.add(level);
        }
        return rst;
    }
}
```

### 107. 二叉树的层次遍历 II

Binary Tree Level Order Traversal II

https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/

```python
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        rst, left = [], []
        if root: left.append(root)
        while left:
            level = []
            for i in range(len(left)):
                node = left.pop(0)
                level.append(node.val)
                if node.left: left.append(node.left)
                if node.right: left.append(node.right)
            rst.insert(0, level)
        return rst
```

```java
public class Solution {
    public ArrayList<ArrayList<Integer>> levelOrderButtom(TreeNode root) {
        // 2015-3-26 BFS
        ArrayList<ArrayList<Integer>> rst = new ArrayList<>();
        if (root == null) {
            return rst;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            ArrayList<Integer> level = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode temp = queue.poll();
                level.add(temp.val);
                if (temp.left != null) {
                    queue.offer(temp.left);
                }
                if (temp.right != null) {
                    queue.offer(temp.right);
                }
            }
            rst.add(0, level);
        } // while
        
        return rst;
    }
}
```

### 103. 二叉树的锯齿形层次遍历

Binary Tree Zigzag Level Order Traversal

https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/

```java
public class Solution {
    public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode root) {
        // 2015-3-26 BFS
        ArrayList<ArrayList<Integer>> rst = new ArrayList<>();
        if (root == null) {
            return rst;
        }
        
        int count = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            ArrayList<Integer> level = new ArrayList<>();
            count++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode temp = queue.poll();
                if (count % 2 == 1) {
                    level.add(temp.val);
                } else {
                    level.add(0, temp.val);
                }
                if (temp.left != null) {
                    queue.offer(temp.left);
                }
                if (temp.right != null) {
                    queue.offer(temp.right);
                }
            }
            rst.add(level);
        }
        return rst;
    }
}
```


### 515. 在每个树行中找最大值

https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row/

```python
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        res = []
        row = []
        if root: row.append(root)
        while row:
            mx = row[0].val
            n = len(row)
            for i in range(n):
                node = row.pop(0)
                mx = max(mx, node.val)
                if node.left: row.append(node.left)
                if node.right: row.append(node.right)
            res.append(mx)
        return res
```

---

## 深度优先遍历 DFS 

前序遍历: 根结点 ---> 左子树 ---> 右子树

中序遍历: 左子树 ---> 根结点 ---> 右子树

后序遍历: 左子树 ---> 右子树 ---> 根结点

![](https://pic.leetcode-cn.com/071065c80aaf44da930c7ccb2156b3eac6309d446eb36a376d6478d17cc2400f-102.png)

## 前序遍历 144. Binary Tree Preorder Traversal

https://leetcode-cn.com/problems/binary-tree-preorder-traversal/

Version 1 递归

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        left = self.preorderTraversal(root.left)
        right = self.preorderTraversal(root.right)
        return [root.val] + left + right
```

```java
public class Solution {
    public ArrayList<Integer> preorderTraversal(TreeNode root) {
        // 2015-3-22 DFS
        ArrayList<Integer> rst = new ArrayList<>();
        if (root == null) {
            return rst;
        }
        // divide
        ArrayList<Integer> left = preorderTraversal(root.left);
        ArrayList<Integer> right = preorderTraversal(root.right);
        // conquer
        rst.add(root.val);
        rst.addAll(left);
        rst.addAll(right);
        return rst;
    }
}
```

Version 2 迭代



## 中序遍历 Binary Tree Inorder Traversal

https://leetcode-cn.com/problems/binary-tree-inorder-traversal/

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        left = self.inorderTraversal(root.left)
        right = self.inorderTraversal(root.right)
        return left + [root.val] + right
```

```java
public class Solution {
    public ArrayList<Integer> inorderTraversal(TreeNode root) {
        // 2015-3-22 DFS
        ArrayList<Integer> rst = new ArrayList<>();
        if (root == null) {
            return rst;
        }
        
        // divide
        ArrayList<Integer> left = inorderTraversal(root.left);
        ArrayList<Integer> right = inorderTraversal(root.right);
        
        // conquer
        rst.addAll(left);
        rst.add(root.val);
        rst.addAll(right);
        
        return rst;
    }
}
```

## 后续遍历 Binary Tree Postorder Traversal

https://leetcode-cn.com/problems/binary-tree-postorder-traversal/

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        left = self.postorderTraversal(root.left)
        right = self.postorderTraversal(root.right)
        return left + right + [root.val]
```

```java
public class Solution {
    public ArrayList<Integer> postorderTraversal(TreeNode root) {
        // 2015-3-22
        ArrayList<Integer> rst = new ArrayList<>();
        if (root == null) {
            return rst;
        }
        
        // divide
        ArrayList<Integer> left = postorderTraversal(root.left);
        ArrayList<Integer> right = postorderTraversal(root.right);
        
        // conquer
        rst.addAll(left);
        rst.addAll(right);
        rst.add(root.val);
        
        return rst;
    }
}
```

## Maximum Depth of Binary Tree

https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/

Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        left_max = self.maxDepth(root.left)
        right_max = self.maxDepth(root.right)
        return max(left_max, right_max) + 1
```

```java
public class Solution {
    public int maxDepth(TreeNode root) {
        // 2015-3-23 DFS
        if (root == null) {
        	return 0;
        }
 
        // divide 
        int left  = maxDepth(root.left);
        int right = maxDepth(root.right);
 
        // conquer
        return Math.max(left, right) + 1;
    }
}
```

## Balanced Binary Tree

[LintCode](https://www.lintcode.com/problem/balanced-binary-tree/description)

Given a binary tree, determine if it is height-balanced.

```python
class Solution:
    def getDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        left = self.getDepth(root.left)
        right = self.getDepth(root.right)
        if left == -1 or right == -1:
            return -1
        if left - right > 1 or right - left > 1:
            return -1
        return max(left, right) + 1
    
    def isBalanced(self, root: TreeNode) -> bool:
        return False if self.getDepth(root) == -1 else True
```

```java
public class Solution {
    public boolean isBalanced(TreeNode root) {
        // 2015-3-23
        return doRecursion(root) != -1;
    }
 
    // return 深度, -1 表示非平衡
    private int doRecursion(TreeNode root) {
    	if (root == null) {
    		return 0;
    	}
 
    	// divide
    	int left = doRecursion(root.left);
    	int right = doRecursion(root.right);
 
    	// conquer
    	if (left == -1 || right == -1) {
    		return -1;
    	}
    	if  (Math.abs(left - right) > 1) {
    		return -1;
    	}
    	return Math.max(left, right) + 1;
    }
}
```

## 236. 二叉树的最近公共祖先

Lowest Common Ancestor

https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        if root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if left and right:
            return root
        if left:
            return left
        if right:
            return right
        return None
        
```

```java
public class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode A, TreeNode B) {
        // 2015-3-23 DFS
        if (root == null || A == null || B == null) {
            return null;
        }
        if (root == A || root == B) {
            return root;
        }
        
        // divide
        TreeNode left = lowestCommonAncestor(root.left, A, B);
        TreeNode right = lowestCommonAncestor(root.right, A, B);
        
        // conquer
        if (left != null && right != null) {
            return root;
        } else if (left != null) {
            return left;
        } else if (right != null) {
            return right;
        } else {
            return null;
        }
    }
}
```

## Construct Binary Tree from Preorder and Inorder Traversal

https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

Given preorder and inorder traversal of a tree, construct the binary tree.

```java
public class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        // 2015-07-22 递归 算清楚左右分支各有几个节点很关键
        if (preorder == null || inorder == null || preorder.length == 0 || inorder.length == 0) {
            return null;
        }
        if (preorder.length != inorder.length) {
            return null;
        }
        return helper(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
    }
    
    // 递归 helper start 和 end 表示子数组首尾元素的序号
    private TreeNode helper(int[] preorder, int prestart, int preend, int[] inorder, int instart, int inend) {
        
        // 叶节点
        if (prestart > preend) {
            return null;
        }
        
        TreeNode root = new TreeNode(preorder[prestart]);
        int pos = findPostion(inorder, instart, inend, preorder[prestart]);
        // 左分支有 pos - instart 个节点
        root.left = helper(preorder, prestart + 1, prestart + pos - instart, inorder, instart, pos - 1);
        // 右分支有 inend - pos 个节点
        root.right = helper(preorder, prestart + pos - instart + 1, preend, inorder, pos + 1, inend);
        return root;
    }
    
    // 返回位置序号
    private int findPostion(int[] arr , int start, int end, int target) {
        if (arr == null || arr.length == 0) {
            return -1;
        }
        for (int i = start; i <= end; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
```

## Binary Tree Maximum Path Sum 

https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/

难度: 3 很有可能会出错

Given a binary tree, find the maximum path sum.
The path may start and end at any node in the tree.

```java
public class Solution {
    private class RstType {
        int maxPath;
        int singlePath;
        public RstType(int s, int m) {
            singlePath = s;
            maxPath = m;
        }
    }
    
    private RstType doRecursion(TreeNode root) {
        if (root == null) {
            return new RstType(0, Integer.MIN_VALUE);
        }
        
        // divide
        RstType left = doRecursion(root.left);
        RstType right = doRecursion(root.right);
        
        // conquer
        int singlePath = Math.max(0,
            root.val + Math.max(left.singlePath, right.singlePath));
        int maxPath = Math.max(Math.max(left.maxPath, right.maxPath),
            root.val + left.singlePath + right.singlePath);
        return new RstType(singlePath, maxPath);
    }
    
    public int maxPathSum(TreeNode root) {
        // 2015-3-23
        RstType rst = doRecursion(root);
        return rst.maxPath;
    }
}
```

---

## Binary Search Tree BST

https://blog.csdn.net/willshine19/article/category/3066953

### 783. Minimum Distance Between BST Nodes

https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/

```python
class Solution:
    def minDiffInBST(self, root: TreeNode) -> int:

        def inorder(node):
            if not node:
                return 
            nonlocal res, pre
            inorder(node.left)
            res = min(res, node.val - pre)
            pre = node.val
            inorder(node.right)
        pre = -99999
        res = 99999

        inorder(root)
        return res
```

### Insert Node in a Binary Search Tree

https://www.lintcode.com/problem/insert-node-in-a-binary-search-tree/description

```java
public class Solution {
    public TreeNode insertNode(TreeNode root, TreeNode node) {
        // 2015-3-30
        if (node == null) {
            return root;
        }
        if (root == null) {
            return node;
        }
        
        TreeNode temp = root;
        TreeNode last = null;
        boolean left = true;
        while (temp != null) {
            if (temp.val < node.val) {
                last = temp;
                temp = temp.right;
                left = false;
            } else if (temp.val > node.val) {
                last = temp;
                temp = temp.left;
                left = true;
            } else {
                return root;
            }
        }
        
        if (left == true) {
            last.left = node;
        } else {
            last.right = node;
        }
        return root;
    }
}
```

### Search Range in Binary Search Tree

https://www.lintcode.com/problem/search-range-in-binary-search-tree/description

```java
public class Solution {
    public ArrayList<Integer> searchRange(TreeNode root, int k1, int k2) {
        // 2015-3-30
        ArrayList<Integer> rst = new ArrayList<>();
        if (root == null) {
            return rst;
        }
        
        // divide
        ArrayList<Integer> left = searchRange(root.left, k1, k2);
        ArrayList<Integer> right = searchRange(root.right, k1, k2);
        
        // conquer
        rst.addAll(left);
        if (k1 <= root.val && k2 >= root.val) {
            rst.add(root.val);
        }
        rst.addAll(right);
        return rst;       
    }
}
```

```java
public class Solution {
    private ArrayList<Integer> results;
    public ArrayList<Integer> searchRange(TreeNode root, int k1, int k2) {
        results = new ArrayList<Integer>();
        helper(root, k1, k2);
        return results;
    }
    
    private void helper(TreeNode root, int k1, int k2) {
        if (root == null) {
            return;
        }
        if (root.val > k1) {
            helper(root.left, k1, k2);
        }
        if (root.val >= k1 && root.val <= k2) {
            results.add(root.val);
        }
        if (root.val < k2) {
            helper(root.right, k1, k2);
        }
    }
}
```

### Remove Node in Binary Search Tree

https://www.lintcode.com/problem/remove-node-in-binary-search-tree/description

```java
public class Solution {
    private TreeNode lastNode;
    private TreeNode targetNode;
 
    private boolean findNode(TreeNode root, int value) {
        while (root != null) {
            if (root.val < value) {
                lastNode = root;
                root = root.right;
            } else if (root.val > value) {
                lastNode = root;
                root = root.left;
            } else {
                targetNode = root;
                return true;// find it
            }
        }
        return false;
    }
    
    private void deleteNode() {
        if (targetNode.left != null && targetNode.right != null) {
            if (lastNode.left == targetNode) {
                lastNode.left = targetNode.left;
            } else {
                lastNode.right = targetNode.left;
            }
                TreeNode temp = targetNode.left;
                while (temp.right != null) {
                    temp = temp.right;
                }
                temp.right = targetNode.right;
        } else if (targetNode.left != null) {
            if (lastNode.left == targetNode) {
                lastNode.left = targetNode.left;
            } else {
                lastNode.right = targetNode.left;
            }
        } else if (targetNode.right != null) {
            if (lastNode.left == targetNode) {
                lastNode.left = targetNode.right;
            } else {
                lastNode.right = targetNode.right;
            }
        } else {
            if (lastNode.left == targetNode) {
                lastNode.left = null;
            } else {
                lastNode.right = null;
            }
        }
    }
    
    public TreeNode removeNode(TreeNode root, int value) {
        // 2015-3-30
        if (root == null) {
            return root;
        }
        TreeNode dummy = new TreeNode(0);
        dummy.left = root;
        // do not find it and do nothing
        if (!findNode(root, value)) {
            return root;
        }
        if (lastNode == null) {
            lastNode = dummy;
        }
        deleteNode();
        
        return dummy.left;
    }
}
// 遇到删除节点的题，一定要考虑，被删除的节点是不是根节点，使用dummyNode
// 使用lastNode时，一定要考虑会不会为空指针
// 待改进
```

### Validate Binary Search Tree

https://www.lintcode.com/problem/validate-binary-search-tree/description

```java
public class Solution {
    /**
     * @param root: The root of binary tree.
     * @return: True if the binary tree is BST, or false
     */
    boolean isFirstNode = true;
    int lastVal;
    public boolean isValidBST(TreeNode root) {
        // 2015-4-1 inoder traversal
        if (root == null) {
            return true;
        }
        
        if (!isValidBST(root.left)) {
            return false;
        }
        
        if (!isFirstNode && lastVal >= root.val) {
            return false;
        }
        isFirstNode = false;
        lastVal = root.val;
        if (!isValidBST(root.right)) {
            return false;
        } 
        
        return true;
    }
}
```

### Convert Sorted List to Binary Search Tree

https://www.lintcode.com/problem/convert-sorted-list-to-binary-search-tree/description

```java
public class Solution {
    public TreeNode sortedListToBST(ListNode head) {  
        // 2015-08-19
        if (head == null) {
            return null;
        }
        if (head.next == null) {
            return new TreeNode(head.val);
        }
        // 至少有两个节点
        // 找到中间节点，左链表和右链表
        ListNode midNode = devideList(head);
        ListNode leftList = head;
        ListNode rightList = midNode.next.next;
        TreeNode root = new TreeNode(midNode.next.val);
        midNode.next = null;
        // 递归 生成树的左右分支
        root.left = sortedListToBST(leftList);
        root.right = sortedListToBST(rightList);
        
        return root;
    }
    
    /**
     * 分割链表，返回前链表的最后一个节点
     * 举例（长度）：
     * 4->2,2 
     * 5->2,3
     */
    private ListNode devideList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        ListNode fast = head.next;
        ListNode slow = head;
        
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        
        return slow;
    }
}
```
这里需要说明一下divideList方法：
类似于findMid，将一个链表分成两个链表，但稍有不同

用法：
mid = divideList(head)
第二个链表mid.next
mid.next = null
第一个链表head

分割实例：
父链 -> 两个自链 -> 数（数字表示链表长度,o表示节点）
0 -> 0,0 -> #
1 -> 1,0 -> o,#,#
2 -> 1,1 -> o,o,#
3 -> 1,2 -> o,o,o
4 -> 2,2 -> o,o,o,o,#,#,#
5 -> 2,3 -> o,o,o,o,#,o,#
6 -> 3,3 -> o,o,o,o,o,o,#
7 -> 3,4 -> o,o,o,o,o,o,o
...


### Inorder Successor in BST

https://www.lintcode.com/problem/inorder-successor-in-bst/description

```java
public class Solution {
    public int searchBigSortedArray(int[] A, int target) {
        // 2015-10-13
        if (A == null || A.length == 0) {
            return -1;
        }
        
        // 优化end以缩小搜索范围
        int end = 0;
        while (end < A.length -1 && A[end] < target) {
            end = end * 2 + 1;
            if (end >= A.length) {
                end = A.length - 1;
            }
        }
        
        // 二分搜索
        int start = 0;
        while (start < end - 1) {
            int mid = start + (end - start) / 2; 
            if (A[mid] >= target) {
                end = mid;
            } else {
                start = mid;
            }
        }
        if (A[start] == target) {
            return start;
        }
        if (A[end] == target) {
            return end;
        }
        return -1;
    }
}
```


## 437. 路径总和 III

https://leetcode-cn.com/problems/path-sum-iii/

## 114. 二叉树展开为链表

https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        if not root: return
        if root.left: self.flatten(root.left)
        if root.right: self.flatten(root.right)
        if not root.left: return

        root.left, root.right = root.right, root.left
        if not root.left: return
        head = root
        while head.right: 
            head = head.right
        head.right, root.left = root.left, head.right
```

## 543. 二叉树的直径

https://leetcode-cn.com/problems/diameter-of-binary-tree/

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.ans = 0
        def depth(r):
            if not r: return 0
            left = depth(r.left)
            right = depth(r.right)
            self.ans = max(self.ans, left + right + 1)
            return max(left, right) + 1
        depth(root)
        return self.ans - 1
```

## 814. 二叉树剪枝

https://leetcode-cn.com/problems/binary-tree-pruning/

```python
class Solution:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        if not root: return root
        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
        if not root.left and not root.right and root.val == 0: 
            return None
        return root
```


## 98. 验证二叉搜索树

https://leetcode-cn.com/problems/validate-binary-search-tree/

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def minMax(root):
            if not root: return []
            left = minMax(root.left)
            right = minMax(root.right)
            if left == None or right == None: return None
            ans = []
            if left:
                if left[-1] >= root.val: return None
                else: ans.append(left[0])
            else: 
                ans.append(root.val)
            if right:
                if right[0] <= root.val: return None
                else: ans.append(right[-1])
            else:
                ans.append(root.val)
            return ans
        return bool(minMax(root) != None)
```

## 101. 对称二叉树

https://leetcode-cn.com/problems/symmetric-tree/

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root: return True
        def isMirror(left, right):
            if not left and not right: return True
            if not left or not right: return False
            return bool(left.val == right.val and isMirror(left.left, right.right) and isMirror(left.right, right.left))
        return isMirror(root.left, root.right)
```