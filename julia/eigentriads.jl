X = zeros(Int, 12, 12*11*10)

n = 1
for p1 in 1:12
    for p2 in 1:12
        if p2 != p1
            for p3 in 1:12
                if p3 != p1 && p3 != p2
                    X[p1, n] = 1
                    X[p2, n] = 1
                    X[p3, n] = 1
                    n = n + 1
                end
            end
        end
    end
end

Cmaj = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
X_fourier = abs.(fft(X, 1))
Cmaj_fourier = abs.(fft(Cmaj, 1))
Cmaj_fourier = repmat(Cmaj_fourier, 1, 12*11*10)
dists_fourier = sum((X_fourier - Cmaj_fourier).*(X_fourier - Cmaj_fourier), 1)
sorted_dists_fourier, sorting_indices_fourier = sort(vec(dists_fourier))
