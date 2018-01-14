using Luxor, Colors

chord = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0];
adjacency = hcat([circshift(chord, p) for p in 0:11] ...);
laplacian = sum(chord) * eye(12) - adjacency;
eigvs = eigvecs(laplacian);


radius = 20
Drawing(16*2*radius, 10*2*radius)

eigv = eigvs[:, 4]
bluered = diverging_palette(240, 10, 200)

origin()
background(bluered[101])
setopacity(1.0)

h = sqrt(3)

xs = [0,  3,  2,  2,  0,  4,  2,  1,  1,  3,  3,  1]
ys = [0,  1,  0,  2,  1,  2,  1,  0,  2,  0,  2,  1]

for p in 1:length(xs)
    c = clamp(imag(eigv[p] * 2), -1, 1)
    c = Int(round(c * 99)) + 101
    setcolor(bluered[c])
    for oct_x in -1:1
        for oct_y in -1:1
            x = xs[p]
            y = ys[p]

            ngon(
                (x + mod(y, 2)/2 + 4oct_x + 1.5*oct_y - 2) * radius * h,
                (y + 3oct_y - 1) * radius * 1.5,
                radius, 6, pi/2, :fill)
            text("C")
        end
    end
end


finish()
preview()
