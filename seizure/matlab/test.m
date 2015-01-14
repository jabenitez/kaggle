
function a = test()

load sample_clip

hold on

plot(data(1,:), 'k')
plot(data(2,:),'r')
plot(data(3,:),'g')
plot(data(4,:),'b')

legend("electrode 1", "electrode 2", "electrode 3", "electrode 4")
ylabel("Electrode Response, Voltage Difference from Mean (V)")
xlabel("x/400 seconds")
a = 4

hold off

print -dpng './output1.png' 
