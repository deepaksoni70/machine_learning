X=[1 2104 5 1 45;1 1416 3 2 40;1 1534 3 2 30;1 852 2 1 36]

y=[460;232;315;178]

disp("X'")
disp(X')

disp("X'X")
disp(X'*X)

disp("pinv(X'X)")
disp(pinv(X'*X))

disp("pinv(X'X)*X'")
disp(pinv(X'*X)*X')

disp("pinv(X'X)*X'*y")
disp(pinv(X'*X)*X'*y)

disp("pinv(X'X)*(X'*y)")
disp(pinv(X'*X)*(X'*y))

