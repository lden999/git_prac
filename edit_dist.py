from typing import List, Tuple

def edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i  
    for j in range(n + 1):
        dp[0][j] = j  

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] 
            else:
                dp[i][j] = 1 + min(
                    dp[i][j - 1],    
                    dp[i - 1][j],    
                    dp[i - 1][j - 1] 
                )
    return dp[m][n]

def spell_check(word: str, dictionary: List[str]) -> Tuple[int, List[str]]:
    min_dist = float('inf')
    suggestions: List[str] = []

    for w in dictionary:
        d = edit_distance(word, w)
        if d < min_dist:
            min_dist = d
            suggestions = [w]
        elif d == min_dist:
            suggestions.append(w)

    return int(min_dist), suggestions

if __name__ == "__main__":
    dictionary = [
        "apple", "banana", "grape", "orange", "mango",
        "pineapple", "apples", "apply", "ape", "grapes"
    ]

    word = input("Enter a word to check spelling: ").strip()
    min_dist, suggestions = spell_check(word, dictionary)

    print(f"\nMinimum Edit Distance: {min_dist}")
    print("Suggested corrections:")
    for s in suggestions:
        print(f" - {s}")
