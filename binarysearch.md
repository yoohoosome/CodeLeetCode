# Binary Search

[toc]

## csdn

https://blog.csdn.net/willshine19/article/category/3088525

## 所有题目

https://leetcode-cn.com/tag/binary-search/

## First Position of Target

https://www.lintcode.com/problem/first-position-of-target/description

```java
class Solution {
    /**
     * @param nums: The integer array.
     * @param target: Target to find.
     * @return: The first position of target. Position starts from 0.
     */
    public int binarySearch(int[] nums, int target) {
        // 2015-4-1 O(log n)
        if (nums == null || nums.length == 0) {
            return -1;
        }
 
        int start = 0;
        int end = nums.length - 1;
        
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (nums[mid] >= target) {
                end = mid;
            } else {
                start = mid;
            }
        } // while
        
        if (nums[start] == target) {
            return start;
        }
        if (nums[end] == target) {
            return end;
        }
        return -1;
    }
}
```

普通的二分搜索，数组中不含重复元素

```java
class Solution {
    /**
     * @param nums: The integer array.
     * @param target: Target to find.
     * @return: The first position of target. Position starts from 0.
     */
    public int binarySearch(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        
        int start = 0;
        int end = nums.length - 1;
        
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (num[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                start = mid + 1;
            }
            if (nums[mid] >= target) {
                end = mid - 1;
            }
        }
        
        if (nums[start] == target) {
            return start;
        }
        if (nums[end] == target) {
            return end;
        }
        return -1;
    }
}
```

python

```python

class Solution:
    def binarySearch(self, nums, target):
        if len(nums) == 0:
            return -1
        
        start = 0
        end = len(nums) - 1
        while start < end - 1:
            mid = (start + end) / 2
            if nums[mid] < target:
                start = mid
            else:
                end = mid
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        return -1

```

go

```go
func binarySearch(nums []int, target int) int {
    if len(nums) == 0 {
        return 0
    }
    
    var left = 0
    var right = len(nums) - 1
    
    if nums[left] >= target {
        return left
    }
    if nums[right] < target {
        return right + 1
    }
    
    for left < right - 1 {
        var mid = (left + right) / 2
        if nums[mid] == target {
            return mid
        } else if nums[mid] > target {
            right = mid
        } else {
            left = mid
        }
    }
    
    if nums[left] == target {
        return left
    } else if nums[right] == target {
        return right
    } else {
        return right
    }
}
```

## 35. 搜索插入位置

https://leetcode-cn.com/problems/search-insert-position/

Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You may assume NO duplicates in the array.

```java
public class Solution {
    /** 
     * param A : an integer sorted array
     * param target :  an integer to be inserted
     * return : an integer
     */
    public int searchInsert(int[] A, int target) {
        // 2015-4-7 binary search
        if (A == null || A.length == 0) {
            return 0;
        }
        
        int start = 0;
        int end = A.length - 1;
        
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (A[mid] >= target) {
                end = mid;
            } else {
                start = mid;
            }
        }
        
        if (A[start] >= target) {
            return start;
        } else if (A[end] >= target) {
            return end;
        } else {
            return end + 1;
        }
    }
}
 
```

python

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if not nums: return -1
        n = len(nums)
        left, right = 0, n - 1

        while left + 1 < right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid
            elif nums[mid] > target:
                right = mid
            else:
                return mid
        
        if nums[left] >= target:
            return left
        elif nums[right] >= target:
            return right
        else:
            return right + 1
```

## Search for a Range

https://www.lintcode.com/problem/search-for-a-range/description

Given a sorted array of n integers, find the starting and ending position of a given target value.

If the target is not found in the array, return [-1, -1].

```java
public class Solution {
    /** 
     *@param A : an integer sorted array
     *@param target :  an integer to be inserted
     *return : a list of length 2, [index1, index2]
     */
    public ArrayList<Integer> searchRange(ArrayList<Integer> A, int target) {
        // 2015-4-6 O(log n)
        ArrayList<Integer> notFind = new ArrayList<>();
        notFind.add(-1);
        notFind.add(-1);
        if (A == null || A.size() == 0) {
            return notFind;
        }
        
        ArrayList<Integer> rst = new ArrayList<>();
        int start = 0;
        int end = A.size() - 1;
        
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (A.get(mid) >= target) {
                end = mid;
            } else {
                start = mid;
            }
        } // while
        
        if (A.get(start) == target) {
            rst.add(start);
        } else if (A.get(end) == target) {
            rst.add(end);
        } else {
            return notFind;
        }
        
        start = 0;
        end = A.size() - 1;
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (A.get(mid) > target) {
                end = mid;
            } else {
                start = mid;
            }
        } // while
        
        if (A.get(end) == target) {
            rst.add(end);
        } else if (A.get(start) == target) {
            rst.add(start);
        } else {
            return notFind;
        }
        return rst;
        
    }
}
 
```

```python
class Solution:
    """
    @param A : a list of integers
    @param target : an integer to be searched
    @return : a list of length 2, [index1, index2]
    """
    def searchRange(self, A, target):
        # write your code here
        if len(A) == 0:
            return [-1, -1]
        start = 0
        end = len(A) - 1
        while start + 1 < end:
            mid = (start + end) / 2
            if A[mid] >= target:
                end = mid
            else:
                start = mid
        if A[start] == target:
            first = start
        elif A[end] == target:
            first = end
        else:
            return [-1, -1]
        
        start = 0
        end = len(A) - 1
        while start + 1 < end:
            mid = (start + end) / 2
            if A[mid] > target:
                end = mid
            else:
                start = mid
        if A[end] == target:
            last = end
        elif A[start] == target:
            last = start
        else:
            return [-1, -1]
        return [first, last]
```

## 74. Search a 2D Matrix

https://leetcode-cn.com/problems/search-a-2d-matrix/

Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.
```
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
```

```java
public class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        // 2015-4-11 binary search
        // 注意矩阵的表示方法
        // 1.行和列都是从0开始计数
        // 2.第一个表示行 第二个表示列 
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        
        int m = matrix.length; //总行数
        int n = matrix[0].length; // 总列数
        int start = 0;
        int end = m * n -1;
        
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (matrix[mid / n][mid % n] > target) {
                end = mid;
            } else if (matrix[mid / n][mid % n] < target) {
                start = mid;
            } else {
                return true;
            }
        }
        
        if (matrix[start / n][start % n] == target) {
            return true;
        }
        if (matrix[end / n][end % n] == target) {
            return true;
        }
        return false;
    }
}
```

```python
'''
2020.2.3
时间 O(nlogn)
'''
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix: return False
        nr = len(matrix)
        if nr == 0: return False;
        nc = len(matrix[0])
        if nc == 0: return False;

        left, right = 0, nr * nc - 1

        def getValue(index):
            return matrix[index // nc][index % nc]

        while left + 1 < right:
            mid = (left + right) // 2
            if getValue(mid) < target:
                left = mid
            elif getValue(mid) > target:
                right = mid
            else:
                return mid
        
        if getValue(left) == target:
            return True
        elif getValue(right) == target:
            return True
        else:
            return False
```
 
## 240. Search a 2D Matrix II

https://leetcode-cn.com/problems/search-a-2d-matrix-ii/

Integers in each row are sorted from left to right.
Integers in each column are sorted from up to bottom.
No duplicate integers in each row or column.

```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```


```java
public class Solution {
    public int searchMatrix(int[][] matrix, int target) {
        // 2019.6.30 O(m*n)
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        
        int m = matrix.length; // 第一层
        int n = matrix[0].length; // 第二层
        
        int rst = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == target) {
                    rst++;
                    break;
                }
            }
        }
        return rst;
    }
}
```

```java
public class Solution {
    public int searchMatrix(int[][] matrix, int target) {
        // 2019.6.30 O(m+n)
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        
        int m = matrix.length; // 第一层
        int n = matrix[0].length; // 第二层
        
        int rst = 0;
        int jStart = n - 1;
        for (int i = 0; i < m; i++) {
            for (int j = jStart; j >= 0; j--) {
                if (matrix[i][j] == target) {
                    rst++;
                    jStart = j - 1;
                    break;
                }
            }
        }
        return rst;
    }
}
```


## First Bad Version

278. 第一个错误的版本

https://leetcode-cn.com/problems/first-bad-version/

```python
class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        if not n: return
        if n == 1: return 1
        start, end = 1, n
        while start < end - 1:
            mid = (start + end) // 2
            if isBadVersion(mid):
                end = mid
            else:
                start = mid
        
        if isBadVersion(start): 
            return start
        else: 
            return end
```


```java
class Solution {
    public int findFirstBadVersion(int n) {
        // 2015-4-11 
        if (n <= 0) {
            return 0;
        }
        
        int start = 1;
        int end = n;
        
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (VersionControl.isBadVersion(mid)) {
                end = mid;
            } else {
                start = mid;
            }
        }
        
        // 有可能第一个version就是BadVersiuon
        if (VersionControl.isBadVersion(start)) {
            return start;
        }
        return end;
        // 不考虑没有BadVersion的情况
    }
}
 
```

## Find Peak Element

https://www.lintcode.com/problem/find-peak-element/description


```java
public class Solution {
    /**
     * @param A: An integers array.
     * @return: return any of peek positions.
     */
    public int findPeak(int[] A) {
        // O(n)
        if (A == null || A.length == 0) {
            return -1;
        }
        
        int peak = A[0];
        int index = 0;
        for (int i = 0; i < A.length; i++) {
            if (peak < A[i]) {
                peak = A[i];
                index = i;
            }
        }
        return index;
    }
}
```

```java
class Solution {
    /**
     * @param A: An integers array.
     * @return: return any of peek positions.
     */
    public int findPeak(int[] A) {
        // 2015-4-12
        if (A == null || A.length <= 2) {
            return -1;
        }
        
        int start = 0;
        int end = A.length - 1;
        
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (A[mid - 1] > A[mid]) {
                end = mid;
            } else if (A[mid] < A[mid + 1]) {
                start = mid;
            } else {
                return mid;
            }
        } // while 肯定能找到
        return -1;
    }
}
 
```

## Search in a Big Sorted Array

```java
public class Solution {
    /**
     * @param A: An integer array
     * @param target: An integer
     * @return : An integer which is the index of the target number
     */
    public int searchBigSortedArray(int[] A, int target) {
        // 2015-10-13 二分搜索的改进
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

---

## Rotated Sorted Array

### Find Minimum in Rotated Sorted Array

4 5 6 7 0 1 2

You can assume no duplicate exists in the array.

https://www.lintcode.com/problem/find-minimum-in-rotated-sorted-array/description

```java
public class Solution {
    /**
     * @param num: a rotated sorted array
     * @return: the minimum number in the array
     */
    public int findMin(int[] num) {
        // 2015-07-19 binary search
        if (num == null || num.length == 0) {
            return 0; // throw exception
        } 
        if (num.length == 1) {
            return num[0];
        }
        
        int start = 0;
        int end = num.length - 1;
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (num[start] < num[end]) {
                return num[start];
            } else {
                if (num[start] < num[mid]) {
                    start = mid;                
                } else {
                    end = mid;
                }
            }
        }
        
        if (num[start] < num[end]) {
            return num[start];
        } else {
            return num[end];
        }
    }    
}
```

### Find Minimum in Rotated Sorted Array II

The array may contain duplicates.

https://www.lintcode.com/problem/find-minimum-in-rotated-sorted-array-ii/description

```java
public class Solution {
    /**
     * @param num: a rotated sorted array
     * @return: the minimum number in the array
     */
    public int findMin(int[] num) {
        // 2015-07-20
        if (num == null || num.length == 0) {
            return 0; // throw exception
        } 
        if (num.length == 1) {
            return num[0];
        }
        
        int start = 0;
        int end = num.length - 1;
        
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (num[start] == num[mid] || num[start] == num[end]) {
                start++;
                continue;
            }
            if (num[end] == num[mid]) {
                end--;
                continue;
            }
            // num[start] num[mid] num[end] 两两不等
            if (num[start] < num[end]) {
                return num[start];
            } else {
                if (num[start] < num[mid]) {
                    start = mid;                
                } else {
                    end = mid;
                }
            }
        }
        
        if (num[start] < num[end]) {
            return num[start];
        } else {
            return num[end];
        }
    }
}
```


### Search in Rotated Sorted Array

https://www.lintcode.com/problem/search-in-rotated-sorted-array/description

```java
public class Solution {
    /** 
     *@param A : an integer rotated sorted array
     *@param target :  an integer to be searched
     *return : an integer
     */
    public int search(int[] A, int target) {
        // 2015-4-7 no duplicate version
        if (A == null || A.length == 0) {
            return -1;
        }
        
        int start = 0;
        int end = A.length - 1;
        
        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (A[end] <= A[start] && A[start] <= A[mid]) {
                if (A[start] <= target && target <= A[mid]) {
                    end = mid;
                } else {
                    start = mid;
                }
            } else if (A[mid] <= A[end] && A[end] <= A[start]) {
                if (A[mid] <= target && target <= A[end]) {
                    start = mid;
                } else {
                    end = mid;
                }
            } else {
                if (A[mid] <= target) {
                    start = mid;
                } else {
                    end = mid;
                }
            }
        }
        if (A[start] == target) {
            return start;
        } else if (A[end] == target) {
            return end;
        } else {
            return -1;
        }
    }
}
 
```

### Search in Rotated Sorted Array II

https://www.lintcode.com/problem/search-in-rotated-sorted-array-ii/description

```java
public class Solution {
    /** 
     * param A : an integer ratated sorted array and duplicates are allowed
     * param target :  an integer to be search
     * return : a boolean 
     */
    public boolean search(int[] A, int target) {
        // 2015-4-7 时间复杂度最快是O(n)
        for (int i = 0; i < A.length; i++) {
            if (A[i] == target) {
                return true;
            }
        }
        return false;
    }
}
 
```
