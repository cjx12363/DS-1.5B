% Test data generation: UMa + UMi scenarios
clc;clear;close all;

out_dir = 'E:\cjx12363\LLM4CP-DS\data\test\';
M_BS = 4; N_BS = 4;
s = qd_simulation_parameters;
s.center_frequency = 2.4e9;

BSAntArray = qd_arrayant.generate('3gpp-mmw',M_BS,N_BS,...
    s.center_frequency,2,7,0.5,1,1,2*0.5,2*0.5);
UEAntArray = qd_arrayant.generate('3gpp-mmw',1,4,...
    s.center_frequency,1,7,0.5,1,1,2*0.5,2*0.5);

scenarios = {'3GPP_38.901_UMa_NLOS', '3GPP_38.901_UMi_NLOS'};
scenario_names = {'UMa', 'UMi'};

for sc = 1:2
    scenario = scenarios{sc};
    sname = scenario_names{sc};
    fprintf('\n=== Test: %s ===\n', sname);

    Speed = 10:10:100;  % 10 speeds
    UENum = 100;         % 100 samples per speed
    H_U_his = zeros(10,100,16,48,4,4,4,2);
    H_U_pre = zeros(10,100,4,48,4,4,4,2);
    H_D_pre = zeros(10,100,4,48,4,4,4,2);

    for iter_Speed = 1:length(Speed)
        tic;
        UESpeed = Speed(iter_Speed);
        Timelength = 19*0.5e-3;
        UETrackLength = UESpeed/3.6*Timelength;

        s1 = qd_simulation_parameters;
        s1.center_frequency = 2.4e9;
        s1.set_speed(UESpeed,0.5e-3);
        s1.use_random_initial_phase = true;
        s1.use_3GPP_baseline = 1;

        BSlocation = [0;0;30];
        rho = 20+30*rand(1,UENum);
        phi = 120*rand(1,UENum)-60;
        UEcenter = [200;0;1.5];
        UElocation = zeros(3,UENum);
        for ind_UE = 1:UENum
            UElocation(:,ind_UE) = [-rho(ind_UE)*cosd(phi(ind_UE)); rho(ind_UE)*sind(phi(ind_UE)); 0]+UEcenter;
        end

        UEtrack = [];
        for ind_UE = 1:UENum
            UEtrack(1,ind_UE) = qd_track.generate('linear',UETrackLength);
            UEtrack(1,ind_UE).name = num2str(ind_UE);
            UEtrack(1,ind_UE).interpolate('distance',1/s1.samples_per_meter,[],[],1);
        end

        l1 = qd_layout(s1);
        l1.no_tx = 1; l1.tx_array = BSAntArray;
        l1.tx_position = BSlocation;
        l1.no_rx = UENum; l1.rx_array = UEAntArray;
        l1.rx_track = UEtrack; l1.rx_position = UElocation;
        l1.set_scenario(scenario);

        [BS2UE_channel,~] = l1.get_channels();
        for ii = 1:UENum
            h = BS2UE_channel(ii).fr(17280e3,96);
            h = reshape(h,2,4,4,4,96,20);
            h = permute(h,[6,5,4,3,2,1]);
            H_U_his(iter_Speed,ii,:,:,:,:,:,:) = h(1:16,1:48,:,:,:,:);
            H_U_pre(iter_Speed,ii,:,:,:,:,:,:) = h(17:20,1:48,:,:,:,:);
            H_D_pre(iter_Speed,ii,:,:,:,:,:,:) = h(17:20,49:96,:,:,:,:);
        end
        fprintf('[%s] Speed %d/10 done (%.0fs)\n', sname, iter_Speed, toc);
    end

    save([out_dir sname '_H_U_his_test.mat'], 'H_U_his', '-v7.3');
    clear H_U_his;
    save([out_dir sname '_H_U_pre_test.mat'], 'H_U_pre', '-v7.3');
    clear H_U_pre;
    save([out_dir sname '_H_D_pre_test.mat'], 'H_D_pre', '-v7.3');
    clear H_D_pre;
    fprintf('=== %s test data saved ===\n', sname);
end
fprintf('\nAll test data saved to %s\n', out_dir);
