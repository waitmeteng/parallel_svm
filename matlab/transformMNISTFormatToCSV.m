function transformMNISTFormatToCSV(file_x, file_y, output_file, num1, num2, tpoints)
    X = loadMNISTImages(file_x);
    X = X';
    y = loadMNISTLabels(file_y);
    y_plus = 0;
    y_minus = 0;
    j = 1;
    M = zeros(tpoints, size(X,2)+1);
    for i = 1:size(y)
        if (y(i) == num1 && y_plus < tpoints/2)
            M(j, 1) = 1;
            M(j, 2:end) = X(i,:);
            y_plus = y_plus + 1; 
            j = j + 1;
        end
        if (y(i) == num2 && y_minus < tpoints/2)
            M(j, 1) = -1;
            M(j, 2:end) = X(i,:);
            y_minus = y_minus + 1;
            j = j + 1;
        end 
    end
    csvwrite(output_file, M);
end