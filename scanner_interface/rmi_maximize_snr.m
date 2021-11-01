function rmi_maximize_snr(img,loc_pars)
global signal_prev;
global fa_prev;
global dfa;
global iter;

% Locations
loc_pars_lck = [loc_pars,'.lck'];

% Do analyzes on current image
iter = iter + 1;
simg = matrix_to_vec(abs(single(img)));
signal = mean(simg)/std(simg);
disp(['   >> iter = ' ,num2str(iter),' and signal = ',num2str(signal)])

% Propagate metric to sequence parameter
if signal > signal_prev(end)
    dfa(end+1) = dfa(end);
else
    dfa(end+1) = -dfa(end);
end

signal_prev(end+1) = signal;
fa_prev(end+1) = fa_prev(end) + dfa(end);
if fa_prev(end) > fa_prev(1)
    fa_prev(end) = fa_prev(1);
end
system(['echo ',num2str(fa_prev(end)),' >> ',loc_pars])
system(['mv ',loc_pars_lck,' ',loc_pars])
disp(['   >> written fa = ',num2str(fa_prev(end))])

figure(777)
subplot(221);plot(signal_prev(2:end));title('SNR approximation')
subplot(222);plot(fa_prev);title('Flip angle [deg]')
subplot(223);imshow(img,[]);title('Complete image')
subplot(224);imshow(reshape(simg,size(img)),[]);title('Analyzed image')
set(gcf,'units','normalized','outerposition',[0 0 1 1],'Color','w')

% END
end