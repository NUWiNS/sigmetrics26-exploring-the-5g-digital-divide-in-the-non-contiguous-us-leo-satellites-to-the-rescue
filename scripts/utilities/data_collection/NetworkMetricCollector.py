

from typing import Dict

class INetworkMetricCollector:
    def get_tcp_dl_stats(self) -> Dict[str, float]:
        raise NotImplementedError
    
    def get_udp_dl_stats(self) -> Dict[str, float]:
        raise NotImplementedError
    
    def get_tcp_ul_stats(self) -> Dict[str, float]:
        raise NotImplementedError
    
    def get_udp_ul_stats(self) -> Dict[str, float]:
        raise NotImplementedError
    
    def get_icmp_rtt_stats(self) -> Dict[str, float]:
        raise NotImplementedError

    def get_statistics(self) -> Dict[str, float]:
        tcp_dl_stats = self.get_tcp_dl_stats()
        udp_dl_stats = self.get_udp_dl_stats()
        tcp_ul_stats = self.get_tcp_ul_stats()
        udp_ul_stats = self.get_udp_ul_stats()
        icmp_rtt_stats = self.get_icmp_rtt_stats()
        
        return {
            'tcp_dl_tput_mbps': tcp_dl_stats,
            'udp_dl_tput_mbps': udp_dl_stats,
            'tcp_ul_tput_mbps': tcp_ul_stats,
            'udp_ul_tput_mbps': udp_ul_stats,
            'icmp_rtt_ms': icmp_rtt_stats
        }
