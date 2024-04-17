# Trie 字典树

### 208. 实现 Trie (前缀树)

https://leetcode-cn.com/problems/implement-trie-prefix-tree/

```python
'''
Trie 前缀树 字典树
'''
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        nxt = self.root
        for c in word:
            if c not in nxt:
                nxt[c] = {}
            nxt = nxt[c]
        nxt["end"] = True 
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        nxt = self.root
        for c in word:
            if c not in nxt: return False
            nxt = nxt[c]
        return "end" in nxt
        
        

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        nxt = self.root
        for c in prefix:
            if c not in nxt: return False
            nxt = nxt[c]
        return True       
```

### 211. 添加与搜索单词 - 数据结构设计

https://leetcode-cn.com/problems/add-and-search-word-data-structure-design/

```python
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        nxt = self.root
        for c in word:
            if c not in nxt:
                nxt[c] = {}
            nxt = nxt[c]
        nxt["end"] = True
            
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        def searchInner(root, word):
            nxt = root
            for i, c in enumerate(word):
                if c == ".":
                    for key in nxt:
                        if key != "end" and searchInner(nxt[key], word[i + 1:]): return True
                    return False
                elif c not in nxt:
                    return False
                nxt = nxt[c]
            if "end" in nxt: return True
        
        return searchInner(self.root, word)
```

146. LRU缓存机制

https://leetcode-cn.com/problems/lru-cache/

![](https://pic.leetcode-cn.com/815038bb44b7f15f1f32f31d40e75c250cec3c5c42b95175ec012c00a0243833-146-1.png)

哈希表 + 双向链表

private:
add_head(node)
remove(node)
pop_tail()

public:
get: remove + add
put: remove + add or add + pop

```python
class DLinkedNode(): 
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None

class LRUCache():
    def add_head(self, node):
        node.prev = self.head
        node.next = self.head.next

        self.head.next.prev = node
        self.head.next = node

    def remove(self, node):
        prev = node.prev
        new = node.next

        prev.next = new
        new.prev = prev


    def pop_tail(self):
        res = self.tail.prev
        self.remove(res)
        return res

    def __init__(self, capacity):
        self.cache = {}
        self.size = 0
        self.capacity = capacity
        self.head, self.tail = DLinkedNode(), DLinkedNode()

        self.head.next = self.tail
        self.tail.prev = self.head
        

    def get(self, key):
        node = self.cache.get(key, None)
        if not node:
            return -1

        # move the accessed node to the head;
        self.remove(node)
        self.add_head(node)

        return node.value

    def put(self, key, value):
        node = self.cache.get(key)

        if not node: 
            newNode = DLinkedNode()
            newNode.key = key
            newNode.value = value

            self.cache[key] = newNode
            self.add_head(newNode)

            self.size += 1

            if self.size > self.capacity:
                # pop the tail
                tail = self.pop_tail()
                del self.cache[tail.key]
                self.size -= 1
        else:
            # update the value.
            node.value = value
            self.remove(node)
            self.add_head(node)
```