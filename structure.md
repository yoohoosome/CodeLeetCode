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
