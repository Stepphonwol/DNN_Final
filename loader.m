%%% read training and testing data into mat
volume = 1024;

raw_all_dirs = dir('.');
all_dirs = [raw_all_dirs(:).isdir];
subFolders = {raw_all_dirs(all_dirs).name}';
subFolders(ismember(subFolders,{'.','..'})) = [];

for i = 1:length(subFolders)
    all_files = dir(fullfile(strcat(subFolders{i}, '/'), '*.png'));
    if strcmp('train', subFolders{i}) == 1
        trainSize = length(all_files);
        trainSrc = zeros(volume, trainSize);
        trainLabels = zeros(1, trainSize);
    elseif strcmp('test', subFolders{i}) == 1
        testSize = length(all_files);
        testSrc = zeros(volume, testSize);
        testLabels = zeros(1, testSize);
    end
    for j = 1:length(all_files)
        image_name = strcat(subFolders{i}, '/', all_files(j).name);
        fprintf("%s", image_name);
        image_label = image_name(end-4);
        image = imread(image_name);
        digits = reshape(im2double(image), 1, volume);
        if strcmp('train', subFolders{i}) == 1
            if strcmp(image_label, '0') == 1
                trainSrc(:,j) = digits;
            elseif strcmp(image_label, '1') == 1
                trainSrc(:,j) = digits;
            end
            trainLabels(j) = str2double(image_label);
        elseif strcmp('test', subFolders{i}) == 1
            if strcmp(image_label, '0') == 1
                testSrc(:,j) = digits;
            elseif strcmp(image_label, '1') == 1
                testSrc(:,j) = digits;
            end
            testLabels(j) = str2double(image_label);
        end
        fprintf("loading %s images %i/%i\n", subFolders{i}, j, length(all_files));
    end
end
save dat.mat trainSize trainSrc trainLabels testSize testSrc testLabels 