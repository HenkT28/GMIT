# Henk Tjalsma, 03-02-2019
# Calculate the factorial of a number.

# start wil be the start number, so we need to keep track of that.
# ans is what the answer will eventually become. It started as 1.
# i stands for iterate, something repetively.

start = 10
ans = 1
i = 1

while i <= start:
  ans = ans * i
  i = i + 1

print(ans)